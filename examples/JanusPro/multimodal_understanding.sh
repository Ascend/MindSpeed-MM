#!/bin/bash
source /usr/local/Ascend/cann/set_env.sh

config_path="./config.json"

python multimodal_understanding.py $config_path