import time

import torch


class InferEngine:
    """Executes preprocessing, generation, and decoding."""

    def __init__(self, args, adapter):
        self.args = args
        self.adapter = adapter

    def run(self, messages: list[dict]) -> dict:
        inputs = self.adapter.preprocess(messages)
        input_token_count = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            started = time.perf_counter()
            outputs = self.adapter.generate(inputs, self.args.inference.generation)
            inference_duration = time.perf_counter() - started
        output_token_count = max(0, int(outputs.shape[-1]) - input_token_count)
        return {
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "inference_duration": inference_duration,
            "output_text": self.adapter.decode(inputs, outputs),
        }
