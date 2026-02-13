git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout fc91372
pip install -e .
pip install triton-ascend==3.2.0
pip install accelerate==1.2.0