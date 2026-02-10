#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.

# . ./path.sh || exit 1;
source /usr/local/Ascend/ascend-toolkit/set_env.sh

stage=0
stop_stage=3

data_dir=./data
pretrained_model_dir=./pretrained_models/Fun-CosyVoice3-0.5B-2512/


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in train-clean-100; do
    mkdir -p data/$x
    python examples/fsdp2/cosyvoice3/preprocess/prepare_data.py --src_dir $data_dir/LibriTTS/$x --des_dir data/$x --instruct "You are a helpful assistant.<|endofprompt|>"
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in train-clean-100; do
    examples/fsdp2/cosyvoice3/preprocess/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in train-clean-100; do
    examples/fsdp2/cosyvoice3/preprocess/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in train-clean-100; do
    mkdir -p data/$x/parquet
    examples/fsdp2/cosyvoice3/preprocess/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --instruct \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi
