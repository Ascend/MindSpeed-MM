import json
from pathlib import Path

from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Configuration parameters for dataset processing
IMAGE_FOLDER = Path("./data/rlhf_v_images/res")  # Directory for storing processed images
OUTPUT_JSON_PATH = "./data/rlhf-v.json"          # Output dataset file (JSON format)
DATASET_NAME = "./data/datasets/rlhf-v"          # Local cache path for Hugging Face dataset


def validate_image(image_path):
    """Quickly validate the integrity of the image file"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False


def process_dataset():
    # Create image storage directory
    IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

    processed_data = []
    dataset = load_dataset(DATASET_NAME)

    for index, item in enumerate(tqdm(dataset['train'], desc="Processing Dataset")):
        try:
            # Data integrity check
            if not all(key in item for key in ['conversations', 'chosen', 'rejected', 'images']):
                raise KeyError("Missing required keys in item")

            # Build data entry
            entry = {
                "messages": [{
                    "role": "user",
                    "content": item['conversations'][0]['value']
                }],
                "chosen": {
                    "role": "assistant",
                    "content": item['chosen']['value']
                },
                "rejected": {
                    "role": "assistant",
                    "content": item['rejected']['value']
                }
            }

            # Process image data
            if not item['images']:
                raise ValueError("Empty images list")

            image_data = item['images'][0].get('bytes')
            if not image_data:
                raise ValueError("Missing image bytes data")

            image_path = IMAGE_FOLDER / f"{index:04d}.jpg"

            # Write binary data directly
            with open(image_path, 'wb') as f:
                f.write(image_data)

            # Validate image validity
            if not validate_image(image_path):
                raise ValueError("Invalid image file")

            entry["images"] = [str(image_path)]
            processed_data.append(entry)

        except Exception as e:
            print(f"\nSkipping item {index} due to error: {str(e)}")
            # Clean up invalid image file
            if 'image_path' in locals() and image_path.exists():
                image_path.unlink()

    # Save processing results
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully processed {len(processed_data)} items. Output saved to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    process_dataset()
