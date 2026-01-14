import os
import json
import time
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLMoeForConditionalGeneration


cls_map = {
    "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
    "Qwen3VLMoeForConditionalGeneration": Qwen3VLMoeForConditionalGeneration
}


def load_inference_data(json_path):
    """
    Load inference dataset from JSON file
    Args:
        json_path: Path to the dataset JSON file
    Returns:
        list: List of inference data, each element is a dictionary containing image_path and text
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file {json_path} does not exist!")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate data format
    if not isinstance(data, list):
        raise ValueError("JSON file content must be in list format!")

    for idx, item in enumerate(data):
        if not isinstance(item, dict) or "image" not in item or "text" not in item:
            raise ValueError(f"Data format error at index {idx}, must contain 'image' and 'text' fields!")

    return data


def validate_model_path(model_path):
    """
    Validate the legality of model path and check necessary files
    Args:
        model_path: Path to the model directory
    Raises:
        FileNotFoundError: If model path or necessary files do not exist
        ValueError: If model path is not a directory
    """
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist!")

    # Check if model path is a directory
    if not os.path.isdir(model_path):
        raise ValueError(f"Model path {model_path} is not a valid directory!")

    # Check if config.json exists in model path
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in model path: {config_path}")


def get_model_type_from_config(model_path):
    """
    Get model type (dense/moe) from config.json in the model directory
    Args:
        model_path: Path to the model directory
    Returns:
        str: Model type, "dense" or "moe"
    Raises:
        KeyError: If "architectures" field not found in config.json
        ValueError: If architecture type is not supported
    """
    # Validate model path first
    validate_model_path(model_path)

    # Load config.json
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Get architectures field
    if "architectures" not in config or len(config["architectures"]) == 0:
        raise KeyError("'architectures' field not found or empty in config.json!")

    architecture = config["architectures"][0]

    model_cls = cls_map[architecture]

    print(f"Automatically detected model type from config.json: (architecture: {architecture})")
    return model_cls


def init_model(model_path):
    """
    Initialize model based on architecture type from config.json
    Args:
        model_path: Path to the model weights directory
    Returns:
        model: Initialized model instance
        processor: Corresponding processor instance
    """
    # Get model type automatically from config.json
    model_cls = get_model_type_from_config(model_path)

    # Load model
    model = model_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False
    ).eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)

    return model, processor


def inference_single_sample(model, processor, image_path, text_prompt, max_new_tokens=1000):
    """
    Perform inference on a single sample
    Args:
        model: Initialized model instance
        processor: Model processor instance
        image_path: Path to the image file
        text_prompt: Text prompt for inference
        max_new_tokens: Maximum number of new tokens to generate
    Returns:
        dict: Dictionary containing inference results and performance metrics
    """
    # Build conversation format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Preprocess input
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt'
    )
    inputs = inputs.to(model.device)

    # Calculate input token count
    input_token_count = inputs.input_ids.size(1)

    # Inference timing
    with torch.no_grad():
        start_time = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        end_time = time.time()

    # Calculate inference metrics
    inference_duration = end_time - start_time
    generated_ids_trimmed = generated_ids[:, len(inputs.input_ids[0]):]
    output_token_count = len(generated_ids_trimmed[0])
    inference_speed = output_token_count / inference_duration if inference_duration > 0 else 0

    # Decode output text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {
        "image_path": image_path,
        "prompt": text_prompt,
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "inference_duration": inference_duration,
        "inference_speed": inference_speed,
        "output_text": output_text
    }


def batch_inference(model_path, data_json_path, max_new_tokens=1000):
    """
    Main function for batch inference
    Args:
        model_path: Path to the model directory
        data_json_path: Path to the inference data JSON file
        max_new_tokens: Maximum number of new tokens to generate
    """
    # 1. Initialize model and processor
    model, processor = init_model(model_path)

    # 2. Load inference data
    print(f"\nLoading inference data from: {data_json_path}")
    inference_data = load_inference_data(data_json_path)
    print(f"Successfully loaded {len(inference_data)} inference samples")

    # 3. Batch inference
    print("\nStarting batch inference...")
    total_duration = 0
    results = []

    for idx, item in enumerate(inference_data):
        print(f"\n===== Processing sample {idx + 1}/{len(inference_data)} =====")
        print(f"Image path: {item['image']}")
        print(f"Prompt: {item['text']}")

        try:
            result = inference_single_sample(
                model, processor,
                item["image"],
                item["text"],
                max_new_tokens
            )
            results.append(result)

            # Print single sample result
            print(f"Input token count: {result['input_token_count']}")
            print(f"Output token count: {result['output_token_count']}")
            print(f"Inference duration: {result['inference_duration']:.4f} seconds")
            print(f"Inference speed: {result['inference_speed']:.2f} tokens/second")
            print(f"Inference result: {result['output_text']}")

            total_duration += result["inference_duration"]

        except Exception as e:
            print(f"Failed to process sample {idx + 1}: {str(e)}")
            continue

    # 4. Print batch inference summary
    print("\n===== Batch Inference Summary =====")
    print(f"Total processed samples: {len(results)}")
    print(f"Total inference duration: {total_duration:.4f} seconds")
    if len(results) > 0:
        avg_speed = sum([r["inference_speed"] for r in results]) / len(results)
        print(f"Average inference speed: {avg_speed:.2f} tokens/second")


if __name__ == "__main__":
    # Configuration parameters
    MODEL_PATH = "./ckpt/Qwen3-VL-30B-A3B-Instruct"  # Model directory path
    DATA_JSON_PATH = "./examples/qwen3vl/infer_demo_data.json"  # Inference dataset path
    MAX_NEW_TOKENS = 1000  # Maximum number of new tokens to generate

    # Execute batch inference
    batch_inference(MODEL_PATH, DATA_JSON_PATH, MAX_NEW_TOKENS)