# Magistral强化学习

## 环境安装
环境搭建及依赖安装
```shell
conda create -n mistral_verl python=3.11
conda activate mistral_verl

pip install torch_npu==2.9.0

source /home/cann/ascend-toolkit/set_env.sh
source /home/cann/nnal/atb/set_env.sh

# 安装vllm
git clone https://github.com/vllm-project/vllm.git
git checkout d7de043d55d1dd629554467e23874097e1c48993
VLLM_TARGET_DEVICE=empty pip install -e .
cd ..

# 安装vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend
git checkout 52d4acfa51fb868823d1070b81cbd2d97e9e4696
pip install -e .
cd ..

# 安装verl
git clone https://github.com/verl-project/verl.git
cd verl
git checkout 4424616d7dfe03cc564866dc5e99dfaba1daba2e
pip install -r requirements.txt
pip install -v -e .
cd ..


# 安装三方库
pip install qwen-vl-utils==0.0.11 mathruler viztracer uvloop==0.21.0 setuptools==80.9.0

# 卸载triton（如有）
pip uninstall triton

# 安装triton-ascend
pip install triton-ascend==3.2.0rc4

# 确保transformers已安装并且版本为4.57.6
pip install transformers==4.57.6

git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```
代码替换：
将verl目录下的verl/utils/vllm/utils.py文件替换为MindSpeed-MM/examples/verl_examples/mistral/utils.py

## 权重下载
- [Magistral-Small-2509](https://huggingface.co/unsloth/Magistral-Small-2509)

## 数据集下载
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)

## 运行
修改`examples/verl_examples/mistral/mistral_lora_grpo.sh`中的cann路径、`data.train_files`、`data.val_files`、` actor_rollout_ref.model.path`、`default_local_dir`参数
运行命令：
```shell
bash examples/verl_examples/mistral/mistral_lora_grpo.sh
```

