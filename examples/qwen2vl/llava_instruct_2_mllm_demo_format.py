import json
import os
import stat

llava_json_path = "./data/llava_instruct_150k.json"
mllm_format_json_path = "./data/mllm_format_llava_instruct_data.json"

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
        img_path = os.path.join("./data/COCO2017/train2017", item["image"])
        print(f"img_path: {img_path}")
        if not os.path.exists(img_path):
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
with os.fdopen(os.open(mllm_format_json_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
    f.write(output_json)
print(f"finish converting dataset into {mllm_format_json_path}")
