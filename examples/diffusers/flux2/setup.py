import os
import sys

from setuptools import find_packages, setup

TARGET_FILES = [
    "./train_dreambooth_lora_flux2.py",
    "./train_dreambooth_lora_flux2_img2img.py",
]
INJECTION_LINE = 110
INJECTION_CODE = """from transformer_patches import apply_patches
apply_patches()

dist.init_process_group(
    backend="hccl",  # Primary backend for NPU
    init_method="env://"
)

if dist.is_initialized():
    cpu_group = dist.new_group(backend="gloo")
else:
    raise RuntimeError("Distributed initialization failed")

from torch._C._distributed_c10d import _register_process_group

_register_process_group("gloo", cpu_group)
"""


def patch_target_files():
    for target in TARGET_FILES:
        if not os.path.exists(target):
            print(
                f"Set up process cannot find {target}. Patch will not be applied.",
                file=sys.stderr,
            )
            continue

        with open(target, "r") as f:
            lines = f.readlines()

        content = "".join(lines)
        if "apply_patches()" in content:
            continue

        insert_index = INJECTION_LINE - 1
        if insert_index > len(lines):
            print(
                f"Warning: {target} has only {len(lines)} lines. Appending at end.",
                file=sys.stderr,
            )
            lines.append("\n" + INJECTION_CODE + "\n")
        else:
            lines.insert(insert_index, INJECTION_CODE + "\n")

        with open(target, "w") as f:
            f.writelines(lines)

        print(f"Patched {target} at line {INJECTION_LINE}.")


# Run patching immediately when setup.py is executed
if __name__ != "distutils.core":
    patch_target_files()

setup(
    name="flux2-patch",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "accelerate==1.10.0",
        "torch==2.7.1",
        "torchvision",
        "transformers==4.55.0",
        "ftfy",
        "tensorboard",
        "Jinja2",
        "peft==0.17.0",
        "sentencepiece",
        "einops",
        "attrs",
        "scipy",
        "decorator",
        "mindstudio-probe",
        "datasets",
    ],
)
