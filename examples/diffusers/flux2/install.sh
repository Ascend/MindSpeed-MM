#!/bin/bash
set -e  # Exit on any error

cd ..
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout 29a930a
cp -r ../MindSpeed-MM/examples/diffusers/flux2/* ./examples/dreambooth
pip install -e .
cd examples/dreambooth
pip install -e .