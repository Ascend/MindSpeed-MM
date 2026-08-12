import time
from typing import Callable, Sequence

import torch
import torch.distributed as dist
from tqdm import tqdm

from mindspeed.fsdp.utils.log import print_rank


class InferEngine:
    """Executes preprocessing, generation, and decoding."""

    def __init__(self, args, adapter):
        self.args = args
        self.adapter = adapter

    def infer(
        self,
        inference_data: Sequence[dict],
        sample_builder: Callable[[dict, int], list[dict]],
    ) -> list[tuple[dict, dict]]:
        total_duration = 0.0
        inference_speeds = []
        inference_results = []
        progress = tqdm(
            inference_data,
            total=len(inference_data),
            desc="Inference",
            unit="sample",
            disable=dist.is_initialized() and dist.get_rank() != 0,
        )

        for index, item in enumerate(progress):
            messages = sample_builder(item, index)
            inputs = self.adapter.preprocess(messages)
            input_token_count = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                started = time.perf_counter()
                outputs = self.adapter.generate(inputs, self.args.inference.generation)
                inference_duration = time.perf_counter() - started
            result = {
                "input_token_count": input_token_count,
                "output_token_count": max(0, int(outputs.shape[-1]) - input_token_count),
                "inference_duration": inference_duration,
                "output_text": self.adapter.decode(inputs, outputs),
            }
            inference_speed = self.infering_log(item, result)
            inference_results.append((item, result))
            total_duration += result["inference_duration"]
            inference_speeds.append(inference_speed)

        print_rank(print, "\n===== Inference Summary =====")
        print_rank(print, f"Total processed samples: {len(inference_data)}")
        print_rank(print, f"Total inference duration: {total_duration:.4f} seconds")
        if inference_speeds:
            average_speed = sum(inference_speeds) / len(inference_speeds)
            print_rank(print, f"Average inference speed: {average_speed:.2f} tokens/second")
        return inference_results

    def infering_log(self, item: dict, result: dict) -> float:
        inference_speed = (
            result["output_token_count"] / result["inference_duration"]
            if result["inference_duration"] > 0
            else 0.0
        )
        print_rank(print, f"\nImage path: {item.get('image', '')}")
        if "videos" in item:
            print_rank(print, f"Video path: {item['videos']}")
        print_rank(print, f"Prompt: {item['text']}")
        print_rank(print, f"Input token count: {result['input_token_count']}")
        print_rank(print, f"Output token count: {result['output_token_count']}")
        print_rank(print, f"Inference duration: {result['inference_duration']:.4f} seconds")
        print_rank(print, f"Inference speed: {inference_speed:.2f} tokens/second")
        print_rank(print, f"Inference result: {result['output_text']}")
        return inference_speed
