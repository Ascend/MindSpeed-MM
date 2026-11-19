# MindSpeed-MM PR Merge Notes

This is an overlay, not a complete MindSpeed-MM checkout. Start from the competition-approved fork at baseline commit `2de94dc0f7453e21015821629036f515673d32e9` and preserve its existing license and third-party notices.

## Merge Contents

1. Copy `examples/qwen3_5/competition_08b/` into the fork at the same relative path. It includes verified remote tooling and the two formal configurations.
2. Keep `competition_evidence/` as the experiment evidence directory.
3. Check and apply the recorded patch from the fork root:

```bash
git apply --check /path/to/mindspeed_mm_pr_overlay/patches/dcp_gloo_process_group.patch
git apply /path/to/mindspeed_mm_pr_overlay/patches/dcp_gloo_process_group.patch
```

Review existing user changes first. Do not overwrite the fork's README or `.gitignore`; merge additions deliberately. `PR_README.md` can be included as PR supporting documentation.

The baseline already contains Qwen3.5 model code. Do not claim upstream model architecture as a newly implemented contribution. The separate project upload directory contains baseline model references in `third_party/mindspeed_mm/`.

## Reproduce

Run launch scripts from the MindSpeed-MM root, because they invoke `mindspeed_mm/fsdp/train/trainer.py` relative to the working directory. Prepare the verified CANN environment, fla_npu operator and Python extension, external model and authorized LLaVA/COCO assets first. Choose new cache/log/checkpoint directories to avoid overwriting prior evidence.

The scripts refer to target-specific `/opt/qwen35_08b/` defaults. Adjust paths before running elsewhere. The scripts folder includes `compare_training_logs.py` beside its wrapper `compare_logs.py`.

## Patch Qualification

The evidence patch was tested for synchronous DCP loading/saving in one-step and formal 100-step two-NPU experiments. It temporarily unbinds the default HCCL group's NPU device when constructing the CPU/Gloo metadata group and restores it afterwards.

The PR patch intentionally leaves the async-save path unchanged because it was not exercised in the formal run. Do not claim async-save compatibility without a separate implementation and test. The original experiment patch remains under `competition_evidence/results/` as immutable evidence of the measured environment.

There is no official precision threshold in this overlay; report loss differences and run the organizer's acceptance script separately. No public fork or PR has been created.
