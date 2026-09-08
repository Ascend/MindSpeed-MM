import time
from typing import Callable

import torch
import torch.distributed as dist
from tqdm import tqdm

from mindspeed.fsdp.utils.log import print_rank


class InferEngine:
    """Executes preprocessing, generation, and decoding."""

    def __init__(self, args, adapter, inference_dataloader, total_samples: int):
        self.args = args
        self.adapter = adapter
        self.inference_dataloader = inference_dataloader
        self.total_samples = total_samples

    def infer(
        self,
        sample_builder: Callable[[dict, int], list[dict]],
    ) -> list[tuple[dict, dict]]:
        inference_results = []
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        progress = tqdm(
            total=self.total_samples,
            desc="Inference",
            unit="sample",
            disable=dist.get_rank() != 0,
        )
        total_started = time.perf_counter()

        for step_index, item in enumerate(self.inference_dataloader):
            global_index = step_index * world_size + global_rank
            is_padding = global_index >= self.total_samples
            global_index = 0 if is_padding else global_index
            messages = sample_builder(item, global_index)
            inputs = self.adapter.preprocess(messages)
            input_token_count = int(inputs["input_ids"].shape[-1])

            with torch.inference_mode():
                started = time.perf_counter()
                outputs = self.adapter.generate(inputs, self.args.inference.generation)
                inference_duration = time.perf_counter() - started

            result = None if is_padding else (
                item,
                {
                    "input_token_count": input_token_count,
                    "output_token_count": max(0, int(outputs.shape[-1]) - input_token_count),
                    "inference_duration": inference_duration,
                    "output_text": self.adapter.decode(inputs, outputs),
                    "global_index": global_index,
                },
            )

            batch_results = self.gather_batch_results(result)
            inference_results.extend(batch_results)
            progress.update(len(batch_results))
            print_rank(print, f"===== Inference Batch {step_index + 1} =====")
            for batch_item, batch_result in batch_results:
                self.infering_log(batch_item, batch_result)

        progress.close()
        total_duration = time.perf_counter() - total_started

        total_input_tokens = sum(result["input_token_count"] for _, result in inference_results)
        total_output_tokens = sum(result["output_token_count"] for _, result in inference_results)

        print_rank(print, "===== Inference Summary =====")
        print_rank(print, f"Total processed samples: {len(inference_results)}")
        print_rank(print, f"Inference rounds: {len(self.inference_dataloader)}")
        print_rank(print, f"Total input tokens: {total_input_tokens}")
        print_rank(print, f"Total output tokens: {total_output_tokens}")
        print_rank(print, f"Total inference duration: {total_duration:.4f} seconds")
        print_rank(print, f"Overall inference speed: {total_output_tokens / total_duration:.2f} tokens/second")

        return inference_results

    @staticmethod
    def gather_batch_results(result: tuple[dict, dict] | None) -> list[tuple[dict, dict]]:
        gathered_results = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(result, gathered_results, dst=0)
        if dist.get_rank() != 0:
            return []

        batch_results = [result for result in gathered_results if result is not None]
        return sorted(batch_results, key=lambda pair: pair[1]["global_index"])

    def infering_log(self, item: dict, result: dict) -> None:
        inference_speed = result["output_token_count"] / result["inference_duration"]

        print_rank(print, f"----- Sample {result['global_index'] + 1} -----")
        if "image" in item:
            print_rank(print, f"Image path: {item['image']}")
        if "video" in item:
            print_rank(print, f"Video path: {item['video']}")
        print_rank(print, f"Prompt: {item['text']}")
        print_rank(print, f"Input token count: {result['input_token_count']}")
        print_rank(print, f"Output token count: {result['output_token_count']}")
        print_rank(print, f"Inference duration: {result['inference_duration']:.4f} seconds")
        print_rank(print, f"Inference speed: {inference_speed:.2f} tokens/second")
        print_rank(print, f"Inference result: {result['output_text']}")
        print_rank(print, "")
