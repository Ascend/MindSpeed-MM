import json
import os

os.environ["USE_TF"] = "FALSE"

import torch.distributed as dist

from mindspeed.fsdp.utils.log import print_rank

from mindspeed_mm.config.config_manager import ConfigManager
from mindspeed_mm.fsdp.evaluation.eval_datasets import eval_dataset_dict
from mindspeed_mm.fsdp.evaluation.eval_impl import eval_impl_dict
from mindspeed_mm.fsdp.inference.inference_runner import InferenceRunner
from mindspeed_mm.fsdp.params.evaluation_args import EvaluationArguments


class EvaluationRunner:
    def __init__(self, args: EvaluationArguments):
        self.inference_runner = InferenceRunner(
            args=args,
            data_provider=self.data_provider,
        )
        evaluator = eval_impl_dict[args.evaluation.dataset_type]
        self.evaluator = evaluator(
            result_output_path=args.evaluation.result_output_path,
            model_name=args.inference.adapter,
            dataset_name=args.evaluation.dataset_type,
        )

    def data_provider(self, args: EvaluationArguments):
        data_builder = eval_dataset_dict[args.evaluation.dataset_type]
        return data_builder(
            args.evaluation.dataset_path,
            max_samples=args.evaluation.max_samples,
        )

    def evaluate(self) -> None:
        inference_results = self.inference_runner.infer()
        if not dist.is_initialized() or dist.get_rank() == 0:
            for item, result in inference_results:
                self.evaluator.update(item, result["output_text"])
            metrics = self.evaluator.finalize()
            print_rank(print, f"Evaluation metrics:\n{json.dumps(metrics, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    arguments = ConfigManager(config_class=EvaluationArguments).load_and_parse()
    evaluation_runner = EvaluationRunner(args=arguments)
    evaluation_runner.evaluate()
