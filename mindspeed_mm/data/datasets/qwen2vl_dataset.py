import os
import warnings
import copy

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers.training_args import TrainingArguments

from megatron.training import get_args
from megatron.training.utils import is_rank0
from mindspeed_mm.data.data_utils.func_utils.convert import (
    DataArguments,
    DataArgumentsForRewardVideo,
    DatasetAttr,
    load_tokenizer,
    align_dataset,
    SupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
)
from mindspeed_mm.data.data_utils.func_utils.log import get_logger
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer
from mindspeed_mm.data.data_utils.video_processor import VideoProcessor
from mindspeed_mm.data.data_utils.video_reader import VideoReader
from mindspeed_mm.data.data_utils.reward_preprocess import build_reward_data

logger = get_logger(__name__)


class DistributedIterableDataset(IterableDataset):
    def __init__(self, dataset, rank=None):
        args = get_args()
        self.data_parallel_size = args.data_parallel_size
        self.dataset = dataset
        self.rank = torch.distributed.get_rank() if rank is None else rank

    def __iter__(self):
        for idx, item in enumerate(self.dataset):
            if idx % self.data_parallel_size == self.rank % self.data_parallel_size:
                yield item


def get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param):
    if "cutoff_len" in basic_param.keys():
        raise ValueError("`cutoff_len` is deprecated, please use `seq_length` instead.")
    data_args = DataArguments(**basic_param)
    data_args.cutoff_len = get_args().seq_length
    process_args = ProcessorArguments(**preprocess_param)
    dataset_attr = DatasetAttr(**dataset_param["attr"])

    tokenizer_module = load_tokenizer(process_args)
    tokenizer, processor = tokenizer_module['tokenizer'], tokenizer_module['processor']
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    # 确保主进程进行数据处理，其他进程复用缓存避免重复计算，该策略和llamafactory对数据处理策略一致
    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # -----------------load dataset from file-------------------------------------------------------------------------
        train_dataset = load_dataset(path="json", data_files=data_args.dataset, split="train",
                                     cache_dir=data_args.cache_dir,
                                     streaming=data_args.streaming)
        if data_args.max_samples and not data_args.streaming:
            train_dataset = train_dataset.select(range(data_args.max_samples))

        val_dataset = None
        if data_args.val_dataset:
            val_dataset = load_dataset(
                path="json",
                data_files=data_args.val_dataset,
                split="train",
                cache_dir=data_args.cache_dir,
                streaming=data_args.streaming
            )
            if data_args.val_max_samples:
                val_dataset = val_dataset.select(range(data_args.val_max_samples))
            if data_args.val_rate is not None and data_args.val_rate > 0.0:
                warnings.warn(
                    "Warning: Both val_dataset and val_rate have been provided. The val_dataset will take priority, and the val_rate will be ignored.",
                    UserWarning)

        local_process_index = int(os.getenv("LOCAL_RANK", -1))
        if data_args.streaming:
            kwargs = {}
        else:
            kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                # 配置了overwrite_cache为false（默认为false)时，非rank0节点读取cache不再进行map处理
                # 配置了overwrite_cache为true（默认为false)时，所有节点都读取cache不再进行map处理
                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
            }
        logger.debug(f'Rank: %s, kwargs: %s', local_process_index, kwargs)
        # -----------------convert to sharegpt ---------------------------------------------------------------------------
        train_dataset = align_dataset(train_dataset, dataset_attr, data_args)
        if val_dataset:
            val_dataset = align_dataset(val_dataset, dataset_attr, data_args)

        # -----------------convert text to token id ----------------------------------------------------------------------
        if dataset_attr.ranking:
            dataset_processor_cls = PairwiseDatasetProcessor
        else:
            dataset_processor_cls = SupervisedDatasetProcessor
        dataset_processor = dataset_processor_cls(template=template, tokenizer=tokenizer, processor=processor,
                                                data_args=data_args)
        preprocess_func = dataset_processor.preprocess_dataset
        if data_args.streaming:
            if data_args.preprocess_on_fly:
                train_dataset.set_transform(
                    preprocess_func,
                    output_all_columns=True,
                )
            else:
                train_dataset = train_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(train_dataset)).keys())),
                    **kwargs,
                )
            train_dataset = DistributedIterableDataset(train_dataset)
            if val_dataset:
                val_dataset = val_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(val_dataset)).keys())),
                    **kwargs,
                )
                val_dataset = DistributedIterableDataset(val_dataset)
                return train_dataset, val_dataset
        else:
            if data_args.preprocess_on_fly:
                train_dataset.set_transform(
                    preprocess_func,
                    output_all_columns=True,
                )
            else:
                train_dataset = train_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(train_dataset)).keys())),
                    desc=f"Rank {local_process_index}, running tokenizer on train_dataset",
                    **kwargs,
                )
            if val_dataset:
                val_dataset = val_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(val_dataset)).keys())),
                    desc=f"Rank {local_process_index}, running tokenizer on val_dataset",
                    **kwargs,
                )
                return train_dataset, val_dataset
        if is_rank0():
            print("training example:")
            dataset_processor.print_data_example(next(iter(train_dataset)))
        return train_dataset


def process_reward_dataset(dataset, data_folder, preprocess_param):
    def add_idx(example, idx):
        example['metainfo_idx'] = idx
        return example

    dataset = dataset.map(lambda example, idx: add_idx(example, idx), with_indices=True)
    if not preprocess_param.get('use_tied_data', True):
        filter_func = lambda example: any(example[f"{dim}"] != "same" for dim in dataset.eval_dim)
        dataset = dataset.filter(filter_func)

    convert_func = lambda example: build_reward_data(example, data_folder, **preprocess_param)
    dataset = dataset.map(convert_func, remove_columns=dataset.column_names)
    return dataset


def get_reward_video_dataset(basic_param, preprocess_param, dataset_param):
    if "cutoff_len" in basic_param.keys():
        raise ValueError("`cutoff_len` is deprecated, please use `seq_length` instead.")

    data_args = DataArgumentsForRewardVideo(**basic_param)

    # 确保主进程进行数据处理，其他进程复用缓存避免重复计算，该策略和llamafactory对数据处理策略一致
    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # -----------------load dataset from file-------------------------------------------------------------------------
        train_dataset = load_dataset(path="csv", data_files=data_args.data_path, split="train",
                                     cache_dir=data_args.cache_dir,
                                     streaming=data_args.streaming)
        if data_args.max_samples and not data_args.streaming:
            train_dataset = train_dataset.select(range(data_args.max_samples))
        
        train_dataset = process_reward_dataset(train_dataset, data_args.data_folder, preprocess_param)
        
        val_dataset = None
        if data_args.val_dataset:
            if data_args.data_path_val:
                val_dataset = load_dataset(path="csv", data_files=data_args.data_path_val, split="train",
                                        cache_dir=data_args.cache_dir,
                                        streaming=data_args.streaming
                )
                val_dataset = process_reward_dataset(val_dataset, data_args.data_folder, preprocess_param)
                train_dataset = train_dataset
            else:
                dataset = train_dataset.train_test_split(test_size=0.02, seed=42)
                train_dataset = dataset['train']
                val_dataset = dataset['test']
            if data_args.val_max_samples:
                val_dataset = val_dataset.select(range(data_args.val_max_samples))
            if data_args.val_rate is not None and data_args.val_rate > 0.0:
                warnings.warn(
                    "Warning: Both val_dataset and val_rate have been provided. The val_dataset will take priority, and the val_rate will be ignored.",
                    UserWarning)
        else:
            train_dataset = train_dataset

        local_process_index = int(os.getenv("LOCAL_RANK", -1))
        if data_args.streaming:
            kwargs = {}
        else:
            kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                # 配置了overwrite_cache为false（默认为false)时，非rank0节点读取cache不再进行map处理
                # 配置了overwrite_cache为true（默认为false)时，所有节点都读取cache不再进行map处理
                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
            }
        logger.debug(f'Rank: %s, kwargs: %s', local_process_index, kwargs)

        if data_args.streaming:
            train_dataset = DistributedIterableDataset(train_dataset)
            if val_dataset:
                val_dataset = DistributedIterableDataset(val_dataset)

        return [train_dataset, val_dataset]