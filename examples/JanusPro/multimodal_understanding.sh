#!/bin/bash
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/Ascend/cann/set_env.sh

config_path="./config.json"

python multimodal_understanding.py $config_path