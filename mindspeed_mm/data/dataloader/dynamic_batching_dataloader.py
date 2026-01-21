# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback

import torch
from torch.utils.data import default_collate

from megatron.training import print_rank_0
from mindspeed_mm.data.dataloader.batching_strategy import TextBatchingStrategy


class DynamicBatchingDataLoader:
    def __init__(
        self,
        dataloader,
        max_seq_len: int,
        dynamic_batch_buffer_size: int,
        drop_last: bool = False,
        vision_layout: str = 'TND',
        consumed_train_samples: int = 0
    ) -> None:
        print_rank_0("[INFO] initializing dynamic batching DataLoader")
        self.vision_layout = vision_layout
        self.dataloader = dataloader
        self.num_step = len(self.dataloader)
        self.batching_strategy = TextBatchingStrategy(
            max_seq_len=max_seq_len,
            buffer_size=dynamic_batch_buffer_size,
        )
        self.drop_last = drop_last
        self.consumed_train_samples = consumed_train_samples
        print_rank_0("[INFO] Successfully initialize dynamic batching DataLoader")

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self.step = 0
        self._data_iter = iter(self.dataloader)
        self._batched_data_iter = self.dynamic_batching_data_generator()
        return self

    def __next__(self):
        return next(self._batched_data_iter)

    def dynamic_batching_data_generator(self):
        while True:
            if self.batching_strategy.is_full_filled():
                micro_batch = self._get_micro_batch()
                yield micro_batch
                self.step += 1
            try:
                processing_item = self._get_data_from_dataloader()
            except Exception as e:
                if isinstance(e, StopIteration):
                    if not self.drop_last and not self.batching_strategy.empty():
                        while not self.batching_strategy.empty():
                            micro_batch = self._get_micro_batch()
                            yield micro_batch
                            self.step += 1
                        return
                    else:
                        return
                else:
                    print(f"DynamicBatchDataset iter data exception: {e} \n{traceback.format_exc()}")
                    raise

            # put processing_item to buffer
            if isinstance(processing_item, dict):
                processing_item = [processing_item]

            for item in processing_item:
                self.batching_strategy.put_item(item)

    def _get_data_from_dataloader(self):
        data = next(self._data_iter)
        data_names = data.keys()

        data['input_ids'] = [data['input_ids'][i][mask] for i, mask in enumerate(data['attention_mask'].bool())]
        data['labels'] = [data['labels'][i][mask] for i, mask in enumerate(data['attention_mask'].bool())]
        data['attention_mask'] = [
            data['attention_mask'][i][mask]
            for i, mask in enumerate(data['attention_mask'].bool())
        ]
        pixel_length = data['image_grid_thw'][:, 0] * data['image_grid_thw'][:, 1] * data['image_grid_thw'][:, 2]
        data['pixel_values'] = data['pixel_values'].split(pixel_length.tolist())

        return [dict(zip(data_names, row)) for row in zip(*data.values())]

    def _get_micro_batch(self):
        micro_batch = self.batching_strategy.get_micro_batch()
        self.consumed_train_samples += len(micro_batch)
        micro_batch = self.collect_fn(micro_batch, self.vision_layout)
        return micro_batch

    @staticmethod
    def collect_fn(features, vision_layout):
        seqlens = torch.tensor([len(feature['input_ids']) for feature in features], dtype=torch.long)
        batch = {"seqlens": seqlens}
        for input_name in features[0].keys():
            if input_name in ('input_ids', 'attention_mask', 'labels'):
                batch[input_name] = torch.cat([feature[input_name] for feature in features]).unsqueeze(0)
            else:
                if input_name == 'pixel_values' and vision_layout == "TND":
                    batch[input_name] = torch.cat([feature[input_name] for feature in features])
                else:
                    batch[input_name] = default_collate([feature[input_name] for feature in features])
        if "attention_mask" in batch.keys():
            batch["indices"] = torch.arange(len(batch["attention_mask"][0]))
        else:
            raise ValueError("Need attention mask to generate indices")
        return batch