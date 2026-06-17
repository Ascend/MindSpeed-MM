import argparse
import json
import os
import stat
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Mock data for VLModels")
    parser.add_argument("--coco_path", type=str, default="./data/COCO2017", help="COCO2017 dataset path")
    parser.add_argument("--llava_json_path", type=str, default="./data/llava_instruct_150k.json", help="Original llava instruct json path")
    parser.add_argument("--output_json_path", type=str, default="./data/mllm_format_llava_instruct_data.json", help="Output mllm format json path")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    llava_json_path = args.llava_json_path
    mllm_format_json_path = args.output_json_path
    with open(llava_json_path, "r") as f:
        info_json = json.loads(f.read())

    mllm_format_llava_instruct_data = []
    for item in info_json:
        # image对应的key或值为空
        if not item.get('image'):
            new_item = {
                "images": [],
                "messages": []
            }
        else:
            img_path = os.path.join("./train2017", item["image"])
            print(f"img_path: {img_path}")
            if not os.path.exists(os.path.join(args.coco_path, img_path)):
                print(f"{os.path.join(args.coco_path, img_path)} does not exit, skipping the sample {img_path}")
                continue
            new_item = {
                "images": [img_path],
                "messages": []
            }

        for i, turn in enumerate(item["conversations"]):
            if turn["from"] == "human":
                new_item["messages"].append({"role": "user", "content": turn["value"]})
            elif turn["from"] == "gpt":
                new_item["messages"].append({"role": "assistant", "content": turn["value"]})
            else:
                raise ValueError(f"unknown role: {turn['from']}")
        mllm_format_llava_instruct_data.append(new_item)

    output_json = json.dumps(mllm_format_llava_instruct_data, ensure_ascii=False)
    if os.path.exists(mllm_format_json_path):
        print(f"{mllm_format_json_path} already exists, please rename it or remove it")

    mllm_format_json_path = Path(mllm_format_json_path)
    # 创建所有不存在的父目录，exist_ok=True表示如果目录已存在不会报错
    mllm_format_json_path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(mllm_format_json_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
        f.write(output_json)
    print(f"finish converting dataset into {mllm_format_json_path}")


"""
e.g.:
python mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py \
    --coco_path ./data/COCO2017 \
    --llava_json_path ./data/llava_instruct_150k.json \
    --output_json_path ./data/mllm_format_llava_instruct_data.json
"""
