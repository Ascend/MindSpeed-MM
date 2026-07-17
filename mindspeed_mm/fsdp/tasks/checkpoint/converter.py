import argparse

from mindspeed_mm.fsdp.checkpoint.convert import build_weight_transform
from mindspeed_mm.fsdp.tasks.checkpoint.utils import hf_to_dcp_sharded, merge_dcp_to_hf_sharded


def hf_to_dcp(
    model_id: str,
    hf_dir: str,
    dcp_dir: str,
    num_workers: int = 0,
) -> None:
    """Convert a Hugging Face checkpoint to DCP with a registered weight pipeline."""
    weight_transform = build_weight_transform(model_id)
    hf_to_dcp_sharded(
        hf_dir=hf_dir,
        dcp_dir=dcp_dir,
        weight_transform=weight_transform,
        num_workers=num_workers,
    )


def dcp_to_hf(
    model_id: str,
    dcp_dir: str,
    hf_dir: str,
    origin_hf_dir: str,
    to_bf16: bool = False,
    num_workers: int = 0,
) -> None:
    """Convert a DCP checkpoint to Hugging Face with a registered weight pipeline."""
    weight_transform = build_weight_transform(model_id)
    merge_dcp_to_hf_sharded(
        load_dir=dcp_dir,
        save_dir=hf_dir,
        model_assets_dir=origin_hf_dir,
        select_key_convert_func=lambda key: f"model.{key}",
        weight_transform=weight_transform,
        to_bf16=to_bf16,
        num_workers=num_workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert checkpoints with a registered weight transform pipeline")
    parser.add_argument("command", choices=("hf_to_dcp", "dcp_to_hf"), help="Conversion direction.")
    parser.add_argument("--model_id", required=True, help="Select the corresponding weight transform pipeline by model ID.")
    parser.add_argument("--hf_dir", required=True, help="HF input directory for hf_to_dcp; HF output directory for dcp_to_hf.")
    parser.add_argument("--dcp_dir", required=True, help="DCP output directory for hf_to_dcp; DCP input directory for dcp_to_hf.")
    parser.add_argument("--origin_hf_dir", help="Original HF assets directory for dcp_to_hf.")
    parser.add_argument("--to_bf16", action="store_true", help="Convert exported HF weights to BF16.")
    parser.add_argument("--num_workers", type=int, default=0, help="Parallel shard workers; 0 means serial.")

    args = parser.parse_args()
    if args.command == "hf_to_dcp":
        hf_to_dcp(
            model_id=args.model_id,
            hf_dir=args.hf_dir,
            dcp_dir=args.dcp_dir,
            num_workers=args.num_workers,
        )
    else:
        if args.origin_hf_dir is None:
            parser.error("--origin_hf_dir is required for dcp_to_hf")
        dcp_to_hf(
            model_id=args.model_id,
            dcp_dir=args.dcp_dir,
            hf_dir=args.hf_dir,
            origin_hf_dir=args.origin_hf_dir,
            to_bf16=args.to_bf16,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
