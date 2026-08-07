import json
import logging
import os

os.environ["USE_TF"] = "FALSE"

import torch
import torch.distributed as dist

from mindspeed.fsdp.utils.log import print_rank, set_log_level
from mindspeed.fsdp.utils.random import set_seed

from mindspeed_mm.config.config_manager import ConfigManager
from mindspeed_mm.fsdp.checkpoint.dcp_checkpointer import DistributedCheckpointer
from mindspeed_mm.fsdp.checkpoint.hf_checkpointer import HuggingFaceCheckpointer
from mindspeed_mm.fsdp.checkpoint.hf_utils import looks_like_hf_weight_dir
from mindspeed_mm.fsdp.checkpoint.utils import retie_embeddings
from mindspeed_mm.fsdp.distributed.parallel_state import init_parallel_state
from mindspeed_mm.fsdp.distributed.torch_parallelize import ParallelApplier
from mindspeed_mm.fsdp.inference.adapters import build_adapter
from mindspeed_mm.fsdp.inference.infer_engine import InferEngine
from mindspeed_mm.fsdp.models.modelhub import ModelHub
from mindspeed_mm.fsdp.params.inference_args import InferenceArguments
from mindspeed_mm.fsdp.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    set_accelerator_compatible,
)
from mindspeed_mm.fsdp.utils.register import import_plugin
from mindspeed_mm.fsdp.utils.utils import to_empty_if_needed


logger = logging.getLogger(__name__)


class InferenceRunner:
    """Builds FSDP2 inference components and manages their lifecycle."""

    def __init__(self, args: InferenceArguments):
        self.args = args
        self.initialize()
        self.model = self.get_model()
        self.adapter = self.get_adapter()
        self.engine = InferEngine(args, self.adapter)

    def initialize(self) -> None:
        """Initialize inference environment: logging, random seeds, distributed groups."""
        args: InferenceArguments = self.args
        print_rank(logger.info, "Start initializing inference environment!!!")

        # Set accelerator compatibility and logging level
        set_accelerator_compatible(get_torch_device())
        set_log_level()
        # Set device index for current process
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.accelerator.set_device_index(local_rank)
        self.device = torch.device(f"{get_device_type()}:{local_rank}")
        # Set random seeds for reproducibility
        set_seed(args.training.seed, set_deterministic=args.training.use_deter_comp)

        # import plugin and trigger register
        import_plugin(getattr(args.training, "plugin", []))

        # Initialize process group for distributed training
        if not dist.is_initialized():
            dist.init_process_group(
                backend=get_dist_comm_backend(cpu=args.parallel.fsdp_plan.cpu_offload),
                device_id=torch.device(f"{get_device_type()}:{local_rank}"),
            )

        # Initialize parallel communication groups and mesh
        init_parallel_state(**args.parallel.to_dict())

    def get_model(self):
        args: InferenceArguments = self.args
        model = ModelHub.build(args.model, args.features, args.training)
        model = ParallelApplier(args.parallel, args.training)(model)

        load_path = args.inference.load
        if args.training.init_model_with_meta_device:
            if not load_path:
                raise ValueError("Meta initialization requires inference.load")
            target_device = "cpu" if args.parallel.fsdp_plan.cpu_offload else get_device_type()
            to_empty_if_needed(model, device=target_device)

        retie_embeddings(model)
        if load_path:
            load_format = args.inference.load_format
            if load_format == "auto":
                load_format = "hf" if looks_like_hf_weight_dir(load_path) else "dcp"
            checkpointer = HuggingFaceCheckpointer if load_format == "hf" else DistributedCheckpointer
            checkpointer.load(
                path=load_path,
                state={"model": model},
                model_id=args.model.model_id,
            )

        model.eval()
        return model

    def get_adapter(self):
        args = self.args
        return build_adapter(
            args.inference.adapter,
            model=self.model,
            processor_path=self.args.inference.processor_path,
            device=self.device,
            param_dtype=args.parallel.fsdp_plan.param_dtype,
            enable_thinking=args.inference.enable_thinking,
        )

    def get_sample(self, item: dict, index: int) -> list[dict]:
        if not isinstance(item.get("text"), str):
            raise ValueError(f"Inference sample {index} must contain a text string")
        image = item.get("image", [])
        if image is None:
            images = []
        elif isinstance(image, str):
            images = [image]
        elif isinstance(image, list) and all(isinstance(path, str) for path in image):
            images = image
        else:
            raise ValueError(f"Inference sample {index} image must be a string or a list of strings")
        videos = item.get("videos", [])
        if videos is None:
            videos = []
        elif isinstance(videos, str):
            videos = [videos]
        elif not isinstance(videos, list) or not all(isinstance(path, str) for path in videos):
            raise ValueError(f"Inference sample {index} videos must be a string or a list of strings")
        messages = [{"type": "image", "value": path} for path in images]
        messages.extend({"type": "video", "value": path} for path in videos)
        messages.append({"type": "text", "value": item["text"]})
        return messages

    def get_samples(self) -> list[dict]:
        args = self.args.inference
        if not os.path.isfile(args.data_path):
            raise FileNotFoundError(f"Inference data file does not exist: {args.data_path}")
        with open(args.data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Inference data file must contain a JSON list")

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Inference sample {index} must be a JSON object")
        return data

    def generate(self):
        inference_data = self.get_samples()
        total_duration = 0.0
        inference_speeds = []

        for index, item in enumerate(inference_data):
            messages = self.get_sample(item, index)
            print_rank(print, f"\n===== Processing sample {index + 1}/{len(inference_data)} =====")
            print_rank(print, f"Image path: {item.get('image', '')}")
            if "videos" in item:
                print_rank(print, f"Video path: {item['videos']}")
            print_rank(print, f"Prompt: {item['text']}")

            result = self.engine.run(messages)
            inference_speed = (
                result["output_token_count"] / result["inference_duration"]
                if result["inference_duration"] > 0
                else 0.0
            )
            print_rank(print, f"Input token count: {result['input_token_count']}")
            print_rank(print, f"Output token count: {result['output_token_count']}")
            print_rank(print, f"Inference duration: {result['inference_duration']:.4f} seconds")
            print_rank(print, f"Inference speed: {inference_speed:.2f} tokens/second")
            print_rank(print, f"Inference result:\n{result['output_text']}")
            total_duration += result["inference_duration"]
            inference_speeds.append(inference_speed)

        print_rank(print, "\n===== Batch Inference Summary =====")
        print_rank(print, f"Total processed samples: {len(inference_data)}")
        print_rank(print, f"Total inference duration: {total_duration:.4f} seconds")
        if inference_speeds:
            print_rank(print, f"Average inference speed: {sum(inference_speeds) / len(inference_speeds):.2f} tokens/second")


if __name__ == "__main__":
    arguments = ConfigManager(config_class=InferenceArguments).load_and_parse()
    inference_runner = InferenceRunner(args=arguments)
    inference_runner.generate()
