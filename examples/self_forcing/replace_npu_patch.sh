# This script is used to place NPU-related patches into their appropriate directories
mkdir -p Self-Forcing/npu_adapt
cp -r examples/self_forcing/npu_adapt Self-Forcing/
echo "Back up the original files: train.py, inference.py, requirements.py, attention.py, self_forcing_dmd.yaml."
mv Self-Forcing/train.py Self-Forcing/train_ori.py
mv Self-Forcing/inference.py Self-Forcing/inference_ori.py
mv Self-Forcing/requirements.txt Self-Forcing/requirements_ori.txt
mv Self-Forcing/configs/self_forcing_dmd.yaml Self-Forcing/configs/self_forcing_dmd_ori.yaml
mv Self-Forcing/wan/modules/attention.py Self-Forcing/wan/modules/attention_ori.py
echo "Replace with NPU-adapted train.py, inference.py, requirements.py, attention.py, self_forcing_dmd.yaml."
cp -r Self-Forcing/npu_adapt/train.py Self-Forcing/
cp -r Self-Forcing/npu_adapt/inference.py Self-Forcing/
cp -r Self-Forcing/npu_adapt/requirements.txt Self-Forcing/
cp -r Self-Forcing/npu_adapt/attention.py Self-Forcing/wan/modules/
cp -r Self-Forcing/npu_adapt/self_forcing_dmd.yaml Self-Forcing/configs/
