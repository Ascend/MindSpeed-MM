# Process the data into the preset format; only the KAN-TTS dataset is supported currently.

import re
import os
import json
import random
import argparse


def clean_text(raw_text):
    pattern = r'#\d+|%'
    cleaned_text = re.sub(pattern, '', raw_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def get_random_wav(wav_folder):
    wav_files = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]
    random_wav = random.choice(wav_files)
    random_wav_path = os.path.join(wav_folder, random_wav)
    return random_wav_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opentts_data_path", type=str, required=True)
    parser.add_argument("--ref_audio_path", type=str, required=False, default=None)
    parser.add_argument("--output_jsonl_path", type=str, required=True)
    args = parser.parse_args()

    wav_folder = os.path.join(args.opentts_data_path, "wav")
    prosody_folder = os.path.join(args.opentts_data_path, "prosody")
    prosody_txt_path = os.path.join(prosody_folder, "prosody.txt")

    if args.ref_audio_path is None or not os.path.exists(args.ref_audio_path):
        ref_audio_path = get_random_wav(wav_folder)
    else:
        ref_audio_path = args.ref_audio_path

    pattern = re.compile(r"^(\d{6})[\t\s]+(.*)$")
    result_list = []
    with open(prosody_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                audio_id = match.group(1)
                raw_text = match.group(2)
                cleaned_text = clean_text(raw_text)
                audio_filename = f"{audio_id}.wav"
                audio_path = os.path.join(wav_folder, audio_filename)
                item = {
                    "audio": audio_path,
                    "text": cleaned_text,
                    "ref_audio": ref_audio_path
                }
                result_list.append(item)

    json_lines = []
    for item in result_list:
        json_str = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
        json_lines.append(json_str)
    final_content = "\n".join(json_lines)
    with open(args.output_jsonl_path, "w", encoding="utf-8") as f:
        f.write(final_content)

if __name__ == "__main__":
    main()