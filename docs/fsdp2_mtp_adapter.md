# FSDP2 通用 MTP 接入指南

MTP（Multi-Token Prediction，多 token 预测）在训练时让模型除了预测下一个 token，还额外预测更后面的 token，并把这部分损失加到总损失里。

“通用 MTP”指不修改模型代码，只在 YAML 中加几行配置，由框架在训练开始前给模型挂上 MTP 模块。本文介绍如何给一个新模型开启通用 MTP。

> 当设置 `mtp_num_layers > 0` 时，若模型自身不含 `mtp` 子模块，框架会自动挂载通用 MTP；若模型代码里已经自带 MTP（模型有 `mtp` 子模块），框架会直接使用模型自带的实现——此时请按该模型的说明配置，不要配置本文第 3 节的 `mtp_adapter`，否则启动时会报错。

## 1. MTP 如何工作

以 `mtp_num_layers: 2` 为例，对序列中的每个位置 t：

| 计算 | 输入 | 预测 | 日志中的损失 |
|---|---|---|---|
| 主模型（原有） | 主干输出的隐藏状态 | 第 t+1 个 token | `loss` |
| MTP 第 1 步 | 主干隐藏状态 + 第 t+1 个 token 的词嵌入 | 第 t+2 个 token | `mtp_1 loss` |
| MTP 第 2 步 | 第 1 步的输出 + 第 t+2 个 token 的词嵌入 | 第 t+3 个 token | `mtp_2 loss` |

每一步先将两路输入归一化并融合投影，经过一个 MTP decoder 层和 Norm 后，再用模型原有的输出头（`lm_head`）计算预测结果。MTP 作为训练期辅助损失使用，只在训练时计算，不影响推理。

## 2. 快速开始

模型结构符合 2.1 节的条件时，只需两步即可开启 MTP。

第一步，在 `model` 下开启 MTP：

```yaml
model:
  mtp_num_layers: 1               # 额外预测 1 个 token（第 t+2 个）；0 表示关闭
  mtp_loss_scaling_factor: 0.1    # MTP 损失的权重，可省略，默认 0.1
```

第二步，如果 `parallel.fsdp_plan.apply_modules` 是按模块列出的，在列表最后加上 `mtp`。框架不会自动添加；漏加时 MTP 参数会并入根模块一起分片，每次前向和反向都要与词嵌入、输出头一起聚合，显存占用更高。

```yaml
parallel:
  fsdp_plan:
    apply_modules:
      # ...原有条目保持不变（例如 - model.layers.{*}）
      - mtp
```

启动训练后，日志中出现 `mtp_1 loss` 即表示 MTP 已生效。

### 2.1 自动识别的模型条件

当模型的子模块按以下标准结构命名时，框架会自动识别并挂载 MTP（可通过 `print(model)` 查看模型结构）：

```text
模型根对象（例如 XxxForCausalLM）
├── model               # 主干，即 lm_head 之前的部分
│   ├── embed_tokens    # 词嵌入
│   ├── layers          # decoder 层列表
│   ├── norm            # 最后一层 Norm
│   └── rotary_emb      # 旋转位置编码（可以没有）
└── lm_head             # 输出头
```

多模态模型中，主干位于 `model.language_model` 或 `language_model.model` 时也能自动识别，输出头位于 `lm_head` 或 `language_model.lm_head` 时同样可以。

除了命名，模型还需要满足以下条件，HuggingFace 风格的实现一般都满足：

- 模型的 forward 通过关键字参数接收 `input_ids` 和 `labels`，返回带 `loss` 字段的输出对象，而不是 tuple。
- 主干的 forward 返回最后一层 Norm 之后的隐藏状态，形状为 `[batch, seq, hidden]`，序列长度与 `input_ids` 相同。返回张量、带 `last_hidden_state` 的对象或第一个元素是张量的 tuple 都可以。
- 模型的 config（或其中的 `text_config`）包含 `hidden_size` 和 `vocab_size`。
- `layers` 中的 decoder 层可以用 `XxxDecoderLayer(config, layer_idx)` 单独创建，forward 只需要隐藏状态和 `attention_mask`、`position_ids`、`position_embeddings` 这类常规参数，不依赖上一层传下来的额外数据。
- `position_ids` 是普通的 `[batch, seq]` 形状。多模态三维位置编码需要额外配置，见 3.4 节。

框架会照着 `layers` 最后一层的类型，新建 MTP decoder 层（默认所有 MTP 步共用同一个层，可通过 `share_decoder: false` 改为每步独立一层），参数与主干不共享。词嵌入和输出头则直接使用主干的模块。

不满足上述条件时，按第 3 节修改配置。

## 3. 模型结构不一致时的配置

以下字段都写在 `model.mtp_adapter` 下，只需要填写与默认值不同的字段。

### 3.1 子模块名称不同

填写对应子模块的路径。**所有路径均从模型根对象开始**，用点号连接属性名。例如代码中访问词嵌入是 `root.text_backbone.embed_tokens`，就写 `text_backbone.embed_tokens`。

```yaml
model:
  mtp_num_layers: 1
  mtp_adapter:
    hidden_state_module: text_backbone    # 主干
    lm_head: output_projection            # 输出头
```

`decoder_layers`、`embed_tokens`、`final_norm` 默认取主干下的 `layers`、`embed_tokens`、`norm`（上例中会自动变为 `text_backbone.layers` 等）。当这些子模块名称也不同时，再单独填写**从模型根对象开始的完整路径**（例如写 `text_backbone.my_layers`，而非只写 `my_layers`）。

### 3.2 Decoder 层无法直接复用

系统默认提取主干网络最后一层（`layers[-1]`）的类来实例化 MTP 层。在接入新模型时，请检查该 Decoder 类的实现，若存在以下情况则需额外配置：

| 检查项 | 需要额外配置的情况 | 配置方式 |
|---|---|---|
| `__init__` 签名 | 除 `config` 与层号外仍有必填参数 | 通过 `layer_kwargs` 补齐 |
| `forward` 的参数与返回值 | 接收或返回层间状态（如跨层残差、上一层的选择结果）；或层内隐藏状态的形状与主干输出的 `[batch, seq, hidden]` 不一致 | 指定 `decoder_class` |
| 层类型随层号变化 | 默认取用 `layers[-1]` 的类，但以 `layer_index`（默认 `0`）实例化。若层类型在 `__init__` 内按层号分支，该组合可能构造出第 0 层的结构；若各层本身属于不同的类，调整 `layer_index` 亦无效 | 同属一个类时通过 `layer_index` 指定层号；分属不同类时指定 `decoder_class` |
| MLP 为 MoE 结构 | 复用后 MTP 的专家位于 `mtp.layers.{*}.mlp.experts`，不在主干 `parallel.ep_plan.apply_modules` 的覆盖范围内，不会被专家并行切分 | 使用 `PlainResidualDecoder` 替换为稠密 MLP，或将 MTP 的专家路径一并加入 `ep_plan.apply_modules` |

模型代码中有合适的层类时，用 `decoder_class` 填写它的完整导入路径：

```yaml
model:
  mtp_num_layers: 1
  mtp_adapter:
    decoder_class: your_package.modeling.YourDecoderLayer
    layer_kwargs:    # 可选，创建层时额外传入的参数
      some_argument: value
```

没有合适的层类时，可以使用框架提供的 `PlainResidualDecoder`。它用模型已有的 Attention、MLP 和 Norm 类组成一个标准的 decoder 层：

```text
h = h + Attention(Norm(h))
h = h + MLP(Norm(h))
```

```yaml
model:
  mtp_num_layers: 1
  mtp_adapter:
    decoder_class: mindspeed_mm.fsdp.models.mtp_decoders.PlainResidualDecoder
    layer_kwargs:
      attention_class: your_package.modeling.YourAttention
      mlp_class: your_package.modeling.YourMLP
      norm_class: your_package.modeling.YourRMSNorm
      config_overrides:    # 可选，只修改 MTP 层使用的 config，不影响主干
        some_config_field: value
```

使用 `PlainResidualDecoder` 时：

- Attention 和 MLP 类需要能用 `Xxx(config)` 或 `Xxx(config, layer_idx)` 创建。
- Norm 类需要能用 `Xxx(hidden_size)`、`Xxx(hidden_size, eps=...)` 或 `Xxx(hidden_size, variance_epsilon=...)` 创建；只接收 `config` 的 Norm 类不支持。
- Attention 返回 tuple 时自动取第一个元素，但 **MLP 必须返回单个张量**，返回 tuple（例如同时返回 router_logits 的 MoE）会因为无法与残差相加而报类型错误。

`layer_index` 只影响新层的实例化，不会复制主干对应层的权重。

完整示例（混合注意力 + 跨层残差状态的模型，使用 `PlainResidualDecoder` 和 `mrope`）见 [Kimi-K3 通用 MTP 配置](../examples/kimi_k3/README.md#5-通用-mtp-配置)。

### 3.3 Attention Mask 格式不匹配

MTP 默认调用 `transformers.masking_utils.create_causal_mask` 生成 4 维因果 Mask 并传入 Decoder 层。若模型 Decoder 层所需的 Mask 格式与此不同，则必须显式配置 `causal_mask_builder`；否则会导致 Mask 语义错误，通常表现为张量维度不匹配或注意力算子内部报错。

**判断与配置方式**：检查主干网络 `forward` 中实际传给 Decoder 层的 `attention_mask` 类型：

- **4 维因果 Mask**：无需额外配置，保持默认即可。
- **2 维布尔 Padding Mask `[batch, seq]`**（常见于线性注意力、稀疏注意力等由层内部自行处理因果掩码的结构）：
  - 若 Decoder 层允许传入 `None`：将 `causal_mask_builder` 设为 `none`。
  - 若 Decoder 层要求 Mask 非空：填入自定义构建函数的导入路径（可参考下方的透传实现）。

```yaml
model:
  mtp_adapter:
    causal_mask_builder: none    # 不生成 4D Mask，由 Decoder 层内部自行处理因果性
```

填入自定义函数路径时，以下关键字参数将被**无条件全部传入**，不会按函数签名过滤，因此该函数必须全部接收或声明 `**kwargs`，否则会报 `TypeError`：

| 关键字 | 实际传入的内容 |
|---|---|
| `config` | MTP 使用的文本 config |
| `inputs_embeds` 或 `input_embeds`（按函数签名取名） | MTP 当前深度移位后重新查表得到的**纯文本**词嵌入，而非主干融合视觉特征后的嵌入 |
| `attention_mask` | 模型 `forward` 收到的原始 `attention_mask` |
| `cache_position`、`past_key_values` | 与主干语义一致 |
| `position_ids` | 剥离多模态轴之后的 **2 维**文本位置，而非原始的 3 维或 4 维 |

返回值将直接作为 `attention_mask` 传入 Decoder 层。若需将原始 Mask 原样透传，可按如下方式实现：

```python
def passthrough_mask(attention_mask=None, **kwargs):
    return attention_mask
```

两点注意：

- **开启序列打包（packing）时该函数不会被调用**，Mask 一律为 `None`，样本边界由 `cu_seqlens` 承担。依赖自定义 Mask 的模型在 packing 场景下需自行处理因果性。
- 关闭 Mask 仅识别小写字符串 `none`。若按 YAML 习惯写成 `null` 或 `~`，将被视为未配置而**静默回退至默认的 4 维因果 Mask**，与预期相反；写成 `None` 则会报 `ValueError`。

### 3.4 多模态位置编码

`position_ids_mode` 根据主干 forward 接收的 `position_ids` 形状选择：

| 取值 | `position_ids` 形状 | 适用模型 |
|---|---|---|
| `text`（默认） | `[batch, seq]` | 普通的一维位置 0、1、2……，纯文本模型通常如此 |
| `mrope` | `[3, batch, seq]` | 多模态模型的三维位置（时间、高、宽，即 M-RoPE）；`[4, batch, seq]`（第一行为文本位置）也会自动处理 |

这个字段只描述 `position_ids` 的形状，与模型是否使用 RoPE 无关。

MTP 使用的 `position_ids` 依次从主干输出的 `position_ids` 属性、主干 forward 的参数、模型 forward 的参数中获取，都没有时按 0、1、2……顺序生成。

### 3.5 不启动训练先验证配置

配置完成后无需占用训练卡即可验证。在训练环境中运行以下代码，可在数十秒内确认模块路径、Decoder 组件与 config 字段能否正常解析：

```python
import json
from types import SimpleNamespace

from accelerate import init_empty_weights

from your_package.configuration_your_model import YourConfig    # 改成你的
from your_package.modeling_your_model import YourForCausalLM    # 改成你的
from mindspeed_mm.fsdp.models.mtp_adapter import attach_mtp_adapter

config = YourConfig(**json.load(open("/path/to/config.json")))
# 实际训练中由 ModelHub 调用 overwrite_transformer_config 注入算子实现等字段，
# 此处需按模型要求手动补齐，否则层可能走到默认分支或因缺少属性报错。

with init_empty_weights():        # 权重建于 meta 设备，不占用显存
    model = YourForCausalLM(config)

model_args = SimpleNamespace(
    mtp_num_layers=1,
    mtp_adapter={...},            # 原样粘贴 yaml 里 mtp_adapter 那一段
)
feature_args = SimpleNamespace(enable_chunk_loss=False, enable_dynamic_chunk_loss=False)

with init_empty_weights():
    model = attach_mtp_adapter(model, model_args, feature_args)
print(model.mtp)
```

请勿为缩短耗时而调小 `num_hidden_layers`：未配置 `decoder_class` 时适配器取用的是 `layers[-1]` 的类，减少层数会导致此处验证到的并非目标层；且许多模型的 config 会校验按层列表的长度与层数一致，减少层数本身即会报错。

可发现的问题：模块路径填写错误、`decoder_class` 与 `layer_kwargs` 的组件构造失败、config 缺少 `hidden_size` 或 `vocab_size`、`causal_mask_builder` 导入失败。无法发现的问题：Mask 语义与张量形状是否正确、显存是否充足，以及 `overwrite_transformer_config` 注入的字段是否齐全，这些需在实际训练中验证。

## 4. 其他训练配置

| 配置 | 说明 |
|---|---|
| 分片 `parallel.fsdp_plan.apply_modules` | 按模块列出时，在最后加上 `mtp`，见第 2 节 |
| 冻结 `model.freeze` | 加入 `mtp` 冻结全部 MTP 参数，加入 `mtp.layers.{*}` 只冻结 MTP 的 decoder 层（仅在加载已有 MTP 权重续训且不希望更新 MTP 时使用；首次开启 MTP 训练时请勿冻结 `mtp`）。词嵌入和输出头与主干共用，需要冻结时按主干中的路径单独配置 |
| ChunkLoss `features.chunkloss_plan.apply_module` | 开启 ChunkLoss 时，填写输出头的路径，与 MTP 使用的 `lm_head` 一致 |
| 权重加载 `training.load_strict` | 基础权重中没有 MTP 参数，框架会自动初始化。保持默认值 `false`；设为 `true` 会因缺少 MTP 参数报错 |
| 显存余量 | MTP 会使单步显存峰值瞬时增加约 1 个 Decoder 层的开销，接入时需确认显存留有余量；具体的释放手段取决于该模型的配置（如序列长度、重计算与激活卸载范围、`features.swap_plan` 的预算） |

- **检查点保存与推理加载**：保存的检查点中，MTP 参数名以 `mtp.` 开头，续训时会正常加载。原模型代码中不含 `mtp` 定义，用 HuggingFace `from_pretrained` 加载时多余的 `mtp.*` 权重会被忽略，不影响常规推理与评测；其他推理引擎如因多余权重报错，转换权重时去掉 `mtp.*` 即可。
- **序列打包与上下文并行**：开启序列打包（packing）时，MTP 的预测不会跨越样本边界；开启 Ulysses CP 时，MTP 会自动按相同方式切分序列。两者都无需额外配置。

## 5. 损失与日志

```text
total loss = loss + 权重 × (mtp_1 loss + ... + mtp_N loss) / N
```

- 权重即 `mtp_loss_scaling_factor`，默认 0.1，可以按需调整。
- `mtp_k loss` 是第 k 步 MTP 在训练中实际算出的损失，不是固定值。
- 例如 `mtp_num_layers: 2`、权重 0.2 时：`total loss = loss + 0.2 × (mtp_1 loss + mtp_2 loss) / 2`。

| 日志字段 | 含义 |
|---|---|
| `loss` | 主模型损失，不含 MTP |
| `mtp_1 loss`、`mtp_2 loss` …… | 每一步 MTP 的损失（未乘权重） |
| `total loss` | 实际用于反向传播的总损失 |

权重设为 0 时 MTP 仍会计算，只是不计入总损失。不需要 MTP 时应设置 `mtp_num_layers: 0`。

## 6. 常见报错

| 报错信息 | 原因 | 处理方法 |
|---|---|---|
| `Model already provides MTP; remove model.mtp_adapter` | 模型自带 MTP | 删除 `mtp_adapter` 配置 |
| `Cannot find the text backbone for MTP` | 主干不在默认位置 | 填写 `hidden_state_module` |
| `Cannot find the output head for MTP` | 输出头不在默认位置 | 填写 `lm_head` |
| `MTP module path '...' does not exist` | 填写的路径或默认推导的路径不存在 | 按报错中的路径检查对应字段（需为从模型根对象开始的完整路径），见 3.1 节 |
| `Cannot construct MTP decoder ...` | 默认的 decoder 层无法用 config 和层号创建 | 填写 `layer_kwargs` 或更换 `decoder_class`，见 3.2 节 |
| `MTP input_ids and text hidden states have different batch/sequence shapes` | 主干输出的序列长度与 `input_ids` 不一致 | 检查 `hidden_state_module` 是否指向文本主干；模型内部会展开 token、导致长度变化的情况暂不支持 |
| `MTP hidden_state_module '...' was not called` | 填写的主干在 forward 中没有被调用 | 改为 forward 中实际调用的模块 |
| `MTP requires return_dict=True from the model` | 模型返回的是 tuple | 让模型返回输出对象 |
| `MTP backbone config has no hidden_size`（或 `vocab_size`） | 默认找到的 config 不是文本 config | 用 `config_path` 指定，例如 `config.language_config` |

## 7. 配置项一览

`model` 下：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `mtp_num_layers` | `0` | 额外预测的 token 数 N，即 MTP 预测步数，`0` 表示关闭 |
| `mtp_loss_scaling_factor` | `0.1` | MTP 损失的权重 |
| `mtp_adapter` | 不配置 | 模型结构与默认规则不同时才需要，字段见下表 |

`model.mtp_adapter` 下（非必填项）：

| 字段 | 默认值 | 何时需要填写 |
|---|---|---|
| `hidden_state_module` | 依次尝试 `model.language_model`、`language_model.model`、`model`，取第一个同时含有 `layers` 和 `embed_tokens` 的模块 | 主干不在这些位置 |
| `decoder_layers` | 主干下的 `layers` | decoder 层列表名称不同（手动填写时需写从模型根对象开始的完整路径） |
| `embed_tokens` | 主干下的 `embed_tokens` | 词嵌入名称不同（手动填写时需写从模型根对象开始的完整路径） |
| `final_norm` | 主干下的 `norm` | 最后一层 Norm 名称不同（手动填写时需写从模型根对象开始的完整路径）；MTP 中的 Norm 使用同一个类 |
| `lm_head` | 依次尝试 `lm_head`、`language_model.lm_head` | 输出头不在这些位置 |
| `rotary_emb` | 主干下的 `rotary_emb`，没有则不使用 | 旋转位置编码在其他位置。主干有而 MTP 层不需要时无法关闭，只能指向一个兼容的模块 |
| `config_path` | 主干的 `config`；**只有主干没有 `config` 时**才回退到模型的 `config.text_config` 或 `config` | 主干的 `config` 存在但 `hidden_size` 在它的 `text_config` 下（该情况不会自动回退，将报缺少 `hidden_size`）。填从模型根对象开始的完整路径，例如 `config.text_config` |
| `decoder_class` | 主干最后一层的类 | 见 3.2 节 |
| `layer_kwargs` | 无 | 创建 decoder 层需要额外参数，见 3.2 节 |
| `layer_index` | `0` | 层的行为与层号有关，见 3.2 节 |
| `share_decoder` | `true` | `true` 时所有 MTP 步共用一个 decoder 层；`false` 时每步各用一个，参数量随 `mtp_num_layers` 增加 |
| `position_ids_mode` | `text` | 多模态三维位置编码时设为 `mrope`，见 3.4 节 |
| `causal_mask_builder` | `transformers.masking_utils.create_causal_mask` | decoder 层要的不是 4 维因果 mask 时填写，见 3.3 节 |
