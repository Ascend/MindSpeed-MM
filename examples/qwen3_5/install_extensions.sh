git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout fc91372
pip install -e .
cd ..

if command -v npu-smi &> /dev/null && npu-smi info &> /dev/null; then
    pip install triton-ascend==3.2.0
fi

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    pip install flash-linear-attention
fi

pip install accelerate==1.2.0