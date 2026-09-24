import argparse
import json
import os

from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import torch
import torch_npu


def get_args():
    parser = argparse.ArgumentParser(description="Mock data for VLModels")
    parser.add_argument("--tokenizer_path", type=str, default="/home/weights/Qwen3.5-35B-A3B/", help="HuggingFace config path")
    parser.add_argument("--pic_width", type=int, default=1024, help="width of the mocked picture")
    parser.add_argument("--pic_height", type=int, default=1024, help="height of the mocked picture")
    parser.add_argument("--num_pics", type=int, default=10, help="number of mocked pictures")
    parser.add_argument("--text_length", type=int, default=16384, help="intended text length of input_ids")
    parser.add_argument("--num_samples", type=int, default=512, help="number of total samples to save")
    parser.add_argument("--save_dir", type=str, default="./data/mocked_vl_data/", help="HuggingFace config path")
    parser.add_argument("--output_format", type=str, default="json", choices=["jsonl", "json"],
                        help="json (default): one whole-array JSON document (small-scale only -- "
                             "1M-scale files over 2GB overflow pyarrow's int32 ReadOptions.block_size on load); "
                             "jsonl: one sample per line -- datasets/pyarrow parses per-line chunks, "
                             "immune to the ~2GB cap; use for large-scale generation")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    path_root = args.save_dir
    num_pics = args.num_pics
    text_length = args.text_length
    json_path = os.path.join(path_root, f"mock_data_pic_num_{num_pics}_textlen_{text_length}.json")
    num_samples = args.num_samples

    # Generate a random RGB image
    width, height = args.pic_width, args.pic_height
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)  # random values in 0-255
    random_image = Image.fromarray(random_array, 'RGB')
    # Save the image
    random_image.save(os.path.join(path_root, f"test_pic_h{height}_w{width}.jpg"))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    seed_text = "A moment frozen in the flow of time"

    repeat_times = text_length // (len(tokenizer.encode(seed_text)) * 2) + 1
    text = (seed_text + ' ') * repeat_times

    image_prompt = "<image>\n" * num_pics

    mock_data = {
        "messages": [
            {
                "content": f"{image_prompt}. Please describe the image. {text}",
                "role": "user"
            },
            {
                "content": text,
                "role": "assistant"
            }
        ],
        "images": [
            f"test_pic_h{height}_w{width}.jpg"
        ] * num_pics
    }
    with open(json_path, 'w', encoding='utf-8') as fw:
        if args.output_format == "jsonl":
            # Write line by line (JSON Lines): the datasets JSON loader parses via
            # pyarrow, whose ReadOptions.block_size is int32 (capped at ~2GB). Dumping
            # the whole array makes pyarrow read the entire file as one batch -- total
            # size over 2GB overflows (OverflowError: value too large to convert to
            # int32_t); line-by-line writing is parsed in per-line chunks, immune to
            # the total size.
            for _ in range(num_samples):
                fw.write(json.dumps(mock_data, ensure_ascii=False) + "\n")
        else:
            # Whole-array JSON document: a single JSON array holding all samples.
            # Only for small-scale debugging -- see the ~2GB pyarrow cap above.
            json.dump([mock_data] * num_samples, fw, ensure_ascii=False)


"""
e.g.:
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SAVE_DIR=./data/mocked_vl_data/
mkdir -p $SAVE_DIR
python mindspeed_mm/fsdp/tools/data_tool/generate_mock_data_for_vlmodel.py \
    --tokenizer_path /home/weights/Qwen3.5-35B-A3B/ \
    --pic_width 1024 \
    --pic_height 1024 \
    --num_pics 10 \
    --text_length 16384 \
    --num_samples 512 \
    --save_dir $SAVE_DIR
"""
