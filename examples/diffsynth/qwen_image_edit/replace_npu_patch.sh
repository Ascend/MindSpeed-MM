# This script is used to replace NPU related patches to their appropriate directories
echo "Back up the original files: README.md, requirements.txt. And replace them."
mv ./README.md ./README_ori.md
mv ./requirements.txt ./requirements_ori.txt
cp ../MindSpeed-MM/examples/diffsynth/qwen_image_edit/README.md ./README.md
cp ../MindSpeed-MM/examples/diffsynth/qwen_image_edit/requirements.txt ./requirements.txt
echo "Move in NPU_adapted scripts."
cp ../MindSpeed-MM/examples/diffsynth/qwen_image_edit/train_qwen_image_edit_lora.sh ./examples/qwen_image/model_training/lora/
cp ../MindSpeed-MM/examples/diffsynth/qwen_image_edit/accelerate_config.yaml ./examples/qwen_image/model_training/lora/
cp ../MindSpeed-MM/examples/diffsynth/qwen_image_edit/inference_qwen_image_edit_bf16.py ./examples/qwen_image/model_inference/
cp ../MindSpeed-MM/examples/diffsynth/qwen_image_edit/inference_qwen_image_edit_lora_bf16.py ./examples/qwen_image/model_inference/
cp ../MindSpeed-MM/examples/diffsynth/qwen_image_edit/qwen_image_edit_patch.py ./diffsynth/models/
