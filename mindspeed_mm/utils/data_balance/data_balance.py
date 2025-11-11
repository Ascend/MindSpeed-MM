import json
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist

from megatron.core import mpu
from megatron.core.parallel_state import get_data_parallel_rank, get_data_parallel_group
from megatron.training import get_args
from mindspeed_mm.utils.data_balance.balance_sorting_algo import SORTING_ALGO_FUNC
from mindspeed_mm.utils.data_balance.batch_generator import PrefetchMicroBatchIterator
from mindspeed_mm.utils.utils import EncoderBalanceComm

TXT_ELEM_SET = {'input_ids', 'attention_mask', 'labels'}
TXT_ELEM_LIST = ['input_ids', 'attention_mask', 'labels']


class DataBalance(object):
    def __init__(
            self,
            virtual_pipeline_model_parallel_size,
            model_config_path,
            sorting_algo_name,
            len_model,
            train_data_iterator
    ):
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        world_size = dist.get_world_size()
        self.state_buffer = {}
        self.sorting_algo = self._get_sorting_algo(sorting_algo_name)
        if get_args().hetero_parallel:
            self.image_encoder_dp = world_size // int(
                model_config['image_encoder']['tp'] *
                model_config['image_encoder']['pp'] *
                model_config['image_encoder']['cp']
            )
        else:
            self.image_encoder_dp = mpu.get_data_parallel_world_size()
        if self.image_encoder_dp <= 1:
            warnings.warn(
                "Data balance is enabled, but the image encoder's data parallelism (dp) is set to 1. "
                "In this case, the data balance feature has no effect and may introduce unnecessary overhead. "
                "Consider disabling it by removing --use-data-balance.",
                UserWarning,
                stacklevel=2
            )
        self.txt_padding_dict = {
            'input_ids':
                train_data_iterator.iterable.gi_frame.f_locals['dl'].collate_fn.data_collator.tokenizer.pad_token_id,
            'labels': train_data_iterator.iterable.gi_frame.f_locals['dl'].collate_fn.data_collator.label_pad_token_id,
            'attention_mask': 0
        }
        self.train_data_iterator = train_data_iterator
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.len_model = len_model

    def build_balanced_train_data_iterator(
            self,
            is_vit_last_stage=False,
            max_batch_capacity=None,
            micro_batch_size=None,
            num_microbatches=None,
            data_type='Unknown data',
            **kwargs
    ):
        batch = next(self.train_data_iterator)
        has_video = 'pixel_values_videos' in batch and 'video_grid_thw' in batch
        if has_video:
            batch['pixel_values'] = batch.pop('pixel_values_videos')
            batch['image_grid_thw'] = batch.pop('video_grid_thw')
        if (mpu.is_pipeline_first_stage() or is_vit_last_stage) and get_args().encoder_dp_balance:
            batch['pixel_values'], batch['tranfer'] = EncoderBalanceComm.apply(
                batch['pixel_values'],
                mpu.get_data_parallel_group())

        split_batch, split_lengths = split_batch_all_data(batch)
        balanced_datas = self.data_balance(
            data_lengths=split_lengths,
            datas=split_batch,
            dp_process_group=get_data_parallel_group(),
            data_type=data_type,
            max_batch_capacity=max_batch_capacity
        )
        balanced_global_batchs = self.get_global_balanced_data(
            balanced_datas,
            micro_batch_size,
            num_microbatches,
            data_type
        )
        micro_batchs = self.collate_fn(balanced_global_batchs)
        if self.virtual_pipeline_model_parallel_size:
            batch_generator = [PrefetchMicroBatchIterator(micro_batchs) for _ in range(self.len_model)]
        else:
            batch_generator = PrefetchMicroBatchIterator(micro_batchs)
        return batch_generator

    def data_balance(
            self,
            data_lengths: torch.Tensor,
            datas: dict[str, torch.Tensor],
            dp_process_group=None,
            data_type='Unknown data',
            max_batch_capacity=None,
            **kwargs):
        dp_rank = get_data_parallel_rank()
        num_replicas = dp_process_group.size()

        gathered_lengths = [
            torch.empty(data_lengths.shape, dtype=data_lengths.dtype, device=data_lengths.device)
            for _ in range(num_replicas)
        ]
        dist.all_gather(gathered_lengths, data_lengths, group=dp_process_group)

        gathered_lengths = torch.stack(gathered_lengths)
        samples_lengths = torch.cat(
            [
                torch.arange(
                    num_replicas, dtype=gathered_lengths.dtype, device=gathered_lengths.device
                ).view(-1, 1).expand(num_replicas, gathered_lengths.shape[1]).unsqueeze(-1),   # group id
                torch.arange(
                    gathered_lengths.shape[1], dtype=gathered_lengths.dtype, device=gathered_lengths.device
                ).view(1, -1).expand(num_replicas, gathered_lengths.shape[1]).unsqueeze(-1),   # batch id
                gathered_lengths
            ], dim=-1
        ).flatten(0, 1)

        rank_table = self.sorting_algo(
            samples_lengths, num_replicas,
            image_encoder_dp=self.image_encoder_dp, max_batch_capacity=max_batch_capacity
        )
        data_list, rank_table = rank_table_mapping(rank_table, dp_rank)

        balance_data_mapping_index = [torch.where(rank_table[dp_rank][:, 0] == r)[0] for r in range(num_replicas)]
        self.state_buffer[data_type] = {
            'balance_data_mapping_index': torch.cat(balance_data_mapping_index),
        }
        balanced_datas = {}
        balanced_data_lengths = torch.empty(
            num_replicas, 2, dtype=rank_table[dp_rank].dtype, device=rank_table[dp_rank].device
        )
        sample_num_per_rank = torch.bincount(rank_table[dp_rank][:, 0], minlength=num_replicas)
        for i, (data_name, data) in enumerate(datas.items()):
            reorganized_data = data_reorganization(data, data_list)
            balanced_data_dim = ()

            if data_name != 'pixel_values':
                balanced_data_dim = (*data[0].shape[1:],)
                balanced_data_lengths[:, 0] = sample_num_per_rank
                balanced_data_lengths[:, 1] = gathered_lengths[:, 0, i]
            else:
                balanced_data_lengths[:, 0] = 0
                balanced_data_lengths[:, 0].index_add_(0, rank_table[dp_rank][:, 0], rank_table[dp_rank][:, 2 + i])
                balanced_data_lengths[:, 1] = data[0].shape[-1]
            balanced_data = all_to_all_communication(
                reorganized_data, balanced_data_lengths, balanced_data_dim, dp_process_group)
            balanced_datas[data_name] = balanced_data

        return balanced_datas

    def get_global_balanced_data(
            self,
            balanced_data_batch: dict,
            micro_batch_size: int,
            num_microbatches: int,
            data_type: str = 'image',
    ):
        split_balanced_data = {}
        split_list = [micro_batch_size] * num_microbatches

        pixel_values = balanced_data_batch.pop('pixel_values')
        image_grid_thws = balanced_data_batch.pop('image_grid_thw')

        # process vision features
        image_grid_thws = torch.cat(image_grid_thws)
        split_grid_data = self.divide_data_based_on_split(image_grid_thws, data_type)
        split_balanced_data['image_grid_thw'] = split_grid_data.split(split_list)

        # origin grid thw to split pixel values
        pixel_split = (image_grid_thws[:, 0] * image_grid_thws[:, 1] * image_grid_thws[:, 2])
        pixel_values_list = torch.cat(pixel_values).split(pixel_split.tolist())
        merge_pixels = [None] * len(pixel_values_list)

        # exchange position
        for i, idx in enumerate(self.state_buffer[data_type]['balance_data_mapping_index']):
            merge_pixels[idx] = pixel_values_list[i]

        # use new grid thw to get batch pixel values
        pixel_split = torch.stack(split_balanced_data['image_grid_thw'])
        pixel_split = (pixel_split[:, :, 0] * pixel_split[:, :, 1] * pixel_split[:, :, 2]).sum(-1)
        split_balanced_data['pixel_values'] = torch.cat(merge_pixels).split(pixel_split.tolist(), dim=0)

        # padding text features
        max_dim = torch.cat([mask.sum(-1) for mask in balanced_data_batch['attention_mask']]).max()
        for new_rank in range(len(balanced_data_batch['attention_mask'])):
            # redundant padding, remove
            if balanced_data_batch['attention_mask'][new_rank].shape[-1] > max_dim:
                balanced_data_batch['input_ids'][new_rank] = balanced_data_batch['input_ids'][new_rank][:, : max_dim]
                balanced_data_batch['labels'][new_rank] = balanced_data_batch['labels'][new_rank][:, : max_dim]
                balanced_data_batch['attention_mask'][new_rank] = (
                    balanced_data_batch['attention_mask'][new_rank][:, : max_dim]
                )
            # padding is not enough, add
            elif balanced_data_batch['attention_mask'][new_rank].shape[-1] < max_dim:
                pad_shape = (0, max_dim - balanced_data_batch['input_ids'][new_rank].size(-1))
                balanced_data_batch['input_ids'][new_rank] = F.pad(
                    balanced_data_batch['input_ids'][new_rank],
                    pad_shape,
                    value=self.txt_padding_dict['input_ids']
                )
                balanced_data_batch['attention_mask'][new_rank] = F.pad(
                    balanced_data_batch['attention_mask'][new_rank],
                    pad_shape,
                    value=self.txt_padding_dict['attention_mask']
                )
                balanced_data_batch['labels'][new_rank] = F.pad(
                    balanced_data_batch['labels'][new_rank],
                    pad_shape,
                    value=self.txt_padding_dict['labels']
                )

        for name, data in balanced_data_batch.items():
            split_data = self.divide_data_based_on_split(torch.cat(data), data_type)
            split_balanced_data[name] = split_data.split(split_list)

        return split_balanced_data

    def divide_data_based_on_split(self, datas, data_type="image") -> torch.Tensor:
        merge_datas = torch.empty_like(datas)
        merge_datas[self.state_buffer[data_type]['balance_data_mapping_index']] = datas
        return merge_datas

    def collate_fn(self, balanced_global_batchs):
        micro_batchs = [dict(zip(balanced_global_batchs.keys(), row)) for row in zip(*balanced_global_batchs.values())]
        return micro_batchs

    def _get_sorting_algo(self, sorting_algo_name):
        return SORTING_ALGO_FUNC[sorting_algo_name]


def split_batch_all_data(datas: dict):
    all_data_batchs = {}
    all_data_lengths = []
    if 'pixel_values' in datas.keys():
        image_grid_thw = datas.pop('image_grid_thw')
        pixel_values = datas.pop('pixel_values')
        pixel_values_length = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
        all_data_batchs['pixel_values'] = pixel_values.npu().split(pixel_values_length.tolist(), dim=0)
        all_data_batchs['image_grid_thw'] = image_grid_thw.npu()

        all_data_lengths.extend([
            pixel_values_length.npu(),
            torch.empty(
                pixel_values_length.shape[0], dtype=torch.long, device='npu'
            ).fill_(image_grid_thw.shape[-1])
        ])

    input_idxs_length = torch.empty(
        datas['input_ids'].shape[0], dtype=torch.long, device='npu'
    ).fill_(datas['input_ids'].shape[-1])

    for key in TXT_ELEM_LIST:
        all_data_batchs[key] = datas.pop(key).npu()
    all_data_lengths.extend([input_idxs_length] * len(TXT_ELEM_LIST))

    for key in datas.keys():
        value = datas[key].npu()
        all_data_batchs[key] = value
        all_data_lengths.extend([
            torch.empty(
                value.shape[0], dtype=torch.long, device=value.device
            ).fill_(value.shape[1] if len(value.shape) > 1 else 1)
        ])

    return all_data_batchs, torch.stack(all_data_lengths, dim=-1)


def all_to_all_communication(data, balanced_data_lengths, data_dim, dp_process_group):
    balanced_data_cache = [
        torch.empty(
            (*new_length, *data_dim), dtype=data[0].dtype, device=data[0].device
        ).squeeze(-1) for new_length in balanced_data_lengths
    ]
    dist.all_to_all(balanced_data_cache, data, group=dp_process_group)
    return balanced_data_cache


def rank_table_mapping(rank_table, dp_rank):
    rank_table = torch.stack(rank_table)
    rank_mask = rank_table[:, :, 0] == dp_rank
    rank_table_for_current_rank = [rank_table[i][rank_mask[i]][:, 1] for i in range(len(rank_table))]

    return rank_table_for_current_rank, rank_table


def data_reorganization(data, data_list):
    if isinstance(data, torch.Tensor):
        new_data_group_per_rank = [data[new_group_idxs] for new_group_idxs in data_list]
    else:
        new_data_group_per_rank = [
            torch.cat([data[idx] for idx in new_group_idxs])
            if new_group_idxs.numel() != 0
            else torch.tensor([], dtype=data[0].dtype, device=data[0].device)
            for new_group_idxs in data_list
        ]

    return new_data_group_per_rank





