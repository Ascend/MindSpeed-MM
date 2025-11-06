import base64
import json
import os
from io import BytesIO

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def trans_base64_to_pil(base64_str: str):
    # 去掉可能的前缀，如"data:image/png;base64"
    if base64_str.startswith('data:image/png;base64,'):
        base64_str = base64_str.split(',')[1]

    # Base64 to bin
    image_data = base64.b64decode(base64_str)

    # bin to PIL
    with Image.open(BytesIO(image_data)) as img:
        img.load()
        return img.copy()


def transform_jsonl(input_path, output_path):
    orig_datas = load_dataset(path=input_path)['train']

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for i, ori_data in enumerate(tqdm(orig_datas)):
            img_data = ori_data['image']
            answer = ori_data['answer']
            img_name = ori_data['image_name']
            img = trans_base64_to_pil(img_data)

            save_path = os.path.join(os.path.dirname(input_path), 'convert')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_file = f"{save_path}/{img_name}.jpg"
            img.save(save_file)

            transformed = {
                'id': i,
                'conversations': [
                    {
                        "role": "<|User|>",
                        "content": "Free OCR.",
                        "images": [f"{save_file}"]
                    },
                    {
                        "role": "<|Assistant|>",
                        "content": answer
                    }
                ]
            }
            outfile.write(json.dumps(transformed, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # user guider
    input_path = './data/CC-OCR' # 替换为实际输入路径
    output_file = './data/output.jsonl' # 替换为实际输出路径

    # run convert
    transform_jsonl(input_path, output_file)

    # for output example
    print("样例输出：")
    with open(output_file, 'r') as f:
        print(json.dumps(json.loads(next(f)), indent=2))