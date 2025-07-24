import base64
import io
import json
import mimetypes
import os
from multiprocessing import Lock

import numpy as np
from PIL import Image

lock = Lock()


def restore_int64_key(obj):
    restored = {}
    for key, value in obj.items():
        if key.isdigit():
            key = np.int64(key)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_value.isdigit():
                    value[sub_key] = int(sub_value)
        restored[key] = value
    return restored


def load_json(pkl_path):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")

    with open(pkl_path, 'r') as f:
        data = json.load(f)

    return restore_int64_key(data)


def save_json(data, pkl_path):
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary.")

    converted_data = {int(k) if isinstance(k, np.integer) else k: v for k, v in data.items()}

    with open(pkl_path, 'w') as f:
        json.dump(converted_data, f)


def is_valid_image(image_path):
    if not os.path.exists(image_path):
        return False
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        if width <= 0 or height <= 0:
            raise ValueError('Image dimensions are invalid')
        return True
    except Exception as e:
        print(f"Error reading image '{image_path}': {e}")
        return False


def parse_file(s):
    if os.path.exists(s) and s != '.':
        if not os.path.isfile(s):
            raise ValueError(f'{s} is not a file')

        suffix = os.path.splitext(s)[1].lower()
        mime = mimetypes.types_map.get(suffix, 'unknown')
        return (mime, s)
    else:
        return (None, s)


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    with lock:
        image = decode_base64_to_image(base64_string, target_size=target_size)
        image.save(image_path)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    with Image.open(io.BytesIO(image_data)) as image:
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        if target_size > 0:
            image.thumbnail((target_size, target_size))
        return image.copy()
