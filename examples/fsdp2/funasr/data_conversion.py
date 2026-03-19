import argparse
import json
import os
import re


def convert_file(input_file, output_file, remote_prefix, local_prefix):
    """
    Convert a single JSONL file by replacing remote URL prefix with local path.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        remote_prefix: Remote URL prefix to be replaced
        local_prefix: Local path prefix to replace with
    """

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
                continue

            # Handle both formats: top-level list or "messages" key
            if isinstance(data, list):
                messages = data
                is_nested = False
            elif "messages" in data:
                messages = data["messages"]
                is_nested = True
            else:
                print(f"Unexpected format: {line}")
                fout.write(line + "\n")
                continue

            # Process each message
            for msg in messages:
                if msg["role"] == "user" and "<|startofspeech|>" in msg["content"]:
                    # Replace the URL prefix
                    msg["content"] = msg["content"].replace(remote_prefix, local_prefix)

            # Write back
            if is_nested:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                fout.write(json.dumps(messages, ensure_ascii=False) + "\n")

    print(f"Converted: {input_file} -> {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FunASR dataset JSONL files by replacing remote URL prefix with local path."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing train_example.jsonl and val_example.jsonl"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--remote_prefix",
        type=str,
        required=True,
        help="Remote URL prefix to be replaced"
    )
    parser.add_argument(
        "--local_prefix",
        type=str,
        default="./funasr-demo",
        help="Local path prefix to replace with"
    )

    args = parser.parse_args()

    # Define input and output file pairs
    file_pairs = [
        ("train_example.jsonl", "train_example_local.jsonl"),
        ("val_example.jsonl", "val_example_local.jsonl")
    ]

    # Process each file pair
    for input_name, output_name in file_pairs:
        input_file = os.path.join(args.input_dir, input_name)
        output_file = os.path.join(args.output_dir, output_name)

        if not os.path.exists(input_file):
            print(f"Warning: Input file not found, skipping: {input_file}")
            continue

        convert_file(input_file, output_file, args.remote_prefix, args.local_prefix)

    print("All conversions completed!")


if __name__ == "__main__":
    main()
