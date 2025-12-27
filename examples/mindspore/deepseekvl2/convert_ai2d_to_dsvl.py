import json
import os


def transform_jsonl(input_path, output_path):
    """处理JSONL文件格式转换

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
            open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # 解析原始数据
            orig_data = json.loads(line.strip())

            # 执行格式转换
            transformed = {
                "id": orig_data["id"],
                "conversations": []
            }

            image_path = orig_data["image"]

            for conv in orig_data["conversations"]:
                role = "<|User|>" if conv["from"] == "human" else "<|Assistant|>"
                new_conv = {
                    "role": role,
                    "content": conv["value"]
                }
                if role == "<|User|>":
                    new_conv["images"] = [image_path]

                transformed["conversations"].append(new_conv)

            # 写入新文件
            outfile.write(json.dumps(transformed, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # 使用示例
    input_file = "input.jsonl"  # 替换为实际输入路径
    output_file = "output.jsonl"  # 替换为实际输出路径

    # 执行转换
    transform_jsonl(input_file, output_file)

    # 验证输出
    print(f"转换完成！输入文件行数: {sum(1 for _ in open(input_file))}")
    print(f"输出文件行数: {sum(1 for _ in open(output_file))}")
    print("样例输出：")
    with open(output_file, 'r') as f:
        print(json.dumps(json.loads(next(f)), indent=2))