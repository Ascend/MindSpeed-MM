# FSDP2框架支持Agentic SFT

## 特性概述

Agentic SFT是一种针对具备工具调用能力的多模态模型的训练方法。本特性在FSDP2框架下支持模型学习如何理解用户意图、调用外部工具并处理工具返回结果，从而实现更智能的多轮对话交互。

## 功能特性

- 支持包含工具调用（tool_call）和工具响应（tool_response）的多轮对话数据训练
- 支持将工具定义（tools schema）注入到系统提示中，使模型能够理解可用的工具能力
- 支持多模态数据（图像、视频、音频）与工具调用的联合训练
- 兼容现有FSDP2训练流程，无需额外配置即可启用

## 数据格式说明

### 数据结构

Agentic SFT数据采用JSON格式，每条数据包含以下关键字段：

```json
{
    "messages": [
        {"role": "system", "content": "系统提示内容"},
        {"role": "user", "content": "用户输入"},
        {"role": "assistant", "content": "助手回复"},
        {"role": "user", "content": "用户输入"},
        {"role": "tool_call", "content": "工具调用请求"},
        {"role": "tool_response", "content": "工具返回结果"},
        {"role": "assistant", "content": "助手基于工具结果的回复"}
    ],
    "audios": "音频文件路径（可选）",
    "images": "图像文件路径（可选）",
    "videos": "视频文件路径（可选）",
    "tools": ["工具定义列表（可选）"]
}
```

### messages 字段详解

`messages` 是核心字段，包含完整的对话历史。每个消息对象包含 `role` 和 `content` 两个属性：

| role 值 | 含义 | 说明 |
|---------|------|------|
| `system` | 系统提示 | 定义助手的角色和行为规范，通常位于messages首位 |
| `user` | 用户输入 | 用户的提问或请求 |
| `assistant` | 助手回复 | 模型的回复内容 |
| `tool_call` | 工具调用 | 模型发起的工具调用请求，使用特殊格式 |
| `tool_response` | 工具响应 | 外部工具返回的结果 |


### tool_call 和 tool_response 的顺序约束

**重要规则：**

1. **交替出现原则**：`工具调用(tool_call)` 和 `工具响应(tool_response)` 必须成对出现，且 `工具调用(tool_call)` 必须在 `工具响应(tool_response)` 之前
2. **位置约束**：工具调用流程通常出现在用户输入之后，形成"用户输入(user) → 工具调用(tool_call) → 工具响应(tool_response) → 助手回复(assistant)"的交互链

## 完整数据示例

```json
[
    {
        "messages": [
            {"role": "system", "content": "你是一名专业、高效的AI智能客服。当前对话的时间为: 2024年06月15日 14时30分00秒 (今天是星期六)。"},
            {"role": "user", "content": "您好，在不在"},
            {"role": "assistant", "content": "您好，请问是杜女士吗？"},
            {"role": "user", "content": "您好，是的，我是杜女士。"},
            {"role": "assistant", "content": "杜女士，感谢您的确认。我是疾控中心的工作人员。我们注意到您还没有接种新型流感疫苗。为了您的健康和公共卫生安全，我们建议您尽快接种。"},
            {"role": "user", "content": "那我选择周三下午三点可以吗？"},
            {"role": "tool_call", "content": "<tool_call>\n{name: register_vaccine_appointment, arguments: {appointment_time: 周三下午三点}}\n送往"},
            {"role": "tool_response", "content": "{status: success, message: 预约成功}"},
            {"role": "assistant", "content": "好的，已经为您登记了周三下午三点的接种时间。请您携带有效身份证件按时前往指定接种点。感谢您对疫情防控工作的支持。"},
            {"role": "user", "content": "谢谢您，辛苦了。"},
            {"role": "assistant", "content": "不客气，这是我们的职责。祝您健康，再见。"}
        ],
        "audios": "/speeches/7_Katerina.wav",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "register_vaccine_appointment",
                    "description": "登记用户的疫苗接种预约", 
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "appointment_time": {
                                "type": "string", 
                                "description": "用户选择的接种时间"
                            }
                        }, 
                        "required": ["appointment_time"]
                    }
                }
            }
        ]
    }
]
```

## 配置说明

### 数据配置

在配置文件中，需要设置 `formatting` 参数为 `multimodal_tool`：

```yaml
data:
  dataset_param:
    attr:
      formatting: multimodal_tool  # 使用 multimodal_tool 格式转换器

    preprocess_parameters:
      template: qwen3_vl_nothink  # 推荐使用 qwen3_vl_nothink 或 qwen3_omni_nothink 模板
      # 如需使用其他模板，可参照 qwen3_vl_nothink 的模板注册代码进行传参 tool_prompt = StringFormatter(slots=[tools_slot])
```

### 支持的模型

当前支持 Agentic SFT 的模型：

- Qwen3.5（推荐使用 `qwen3_vl_nothink` 模板）
- Qwen3Omni（推荐使用 `qwen3_omni_nothink` 模板）

## 注意事项

1. **数据校验**：训练前请确保 `tool_call` 和 `tool_response` 正确配对，未配对的工具调用会导致数据被跳过
2. **模板兼容**：确保选择的模板支持工具调用格式，目前 `qwen3_vl_nothink` 模板已完整支持
3. **多模态支持**：Agentic SFT 支持与图像、视频、音频数据联合训练，但需确保文件路径正确
4. **工具定义**：`tools` 字段可选，如需注入工具定义到系统提示，请按规范格式填写
