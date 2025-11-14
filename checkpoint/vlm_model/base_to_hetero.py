import os 
import argparse
from pathlib import Path
from tqdm import tqdm
import torch


def convert_base_to_hetero(input_dir, output_dir):
    iter_name = "release"

    def should_copy(key):
        encoder_prefixes = ["image_encoder.", "audio_encoder."]
        return key.startswith(tuple(encoder_prefixes))

    input_iter_dir = Path(input_dir, iter_name)
    output_iter_dir = Path(output_dir, iter_name)
    

    rank0_path = input_iter_dir / "mp_rank_00_000" / "model_optim_rng.pt"
    rank0_data = torch.load(rank0_path)

    encoder_weights = {
        k: v
        for k, v in rank0_data['model'].items()
        if should_copy(k)
    }

    output_iter_dir.mkdir(parents=True, exist_ok=True)

    all_ranks = [d for d in input_iter_dir.iterdir() if d.is_dir() and d.name.startswith("mp_rank_")]

    print(f"Copying encoder weights to {len(all_ranks)} rank directories...")
    with tqdm(total=len(all_ranks), desc="Processing Ranks", unit="rank") as pbar:
        for rank_dir in all_ranks:
            rank_base = os.path.basename(rank_dir)
            input_rank_path = input_iter_dir / rank_base / "model_optim_rng.pt"
            output_rank_path = output_iter_dir / rank_base / "model_optim_rng.pt"
            output_rank_path.parent.mkdir(parents=True, exist_ok=True)
            data = torch.load(input_rank_path, map_location='cpu')

            injected_count = 0
            for k, v in encoder_weights.items():
                if k not in data['model']:
                    data['model'][k] = v
                    injected_count += 1
            torch.save(data, output_rank_path)
            pbar.update(1)

    Path(output_dir, "latest_checkpointed_iteration.txt").write_text(iter_name)
    print(f"Completed. Output directory: {output_dir}")