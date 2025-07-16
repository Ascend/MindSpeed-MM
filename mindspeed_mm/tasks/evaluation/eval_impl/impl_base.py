import os
import string

import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm
from megatron.core import mpu

from mindspeed_mm.tasks.evaluation.utils.analysis_utils import mmmu_open_question_preprocess, report_acc, eval_vanilla, \
    track_progress_rich
from mindspeed_mm.tasks.evaluation.utils.string_utils import logger_rank_0
from mindspeed_mm.tasks.evaluation.utils.file_utils import load_json, save_json, save_csv
from mindspeed_mm.tasks.evaluation.eval_prompt.build_prompt_base import BasePromptTemplate
from mindspeed_mm.tasks.evaluation.eval_datasets.datasets_base import BaseEvalDataset


class BaseEvalImpl:

    def __init__(self, dataset: BaseEvalDataset, inference_pipeline, args, model_prompt_template=None, drop_last=False):

        self.rank = mpu.get_data_parallel_group().rank()
        self.world_size = mpu.get_data_parallel_world_size()

        self.output_path = args.result_output_path
        os.makedirs(self.output_path, exist_ok=True)
        model_name = args.evaluation_model
        self.dataset = dataset
        self.inference_pipeline = inference_pipeline
        self.model_prompt_template = model_prompt_template

        self.dataset_name = args.evaluation_dataset
        self.supported_types = ['text', 'image', 'video']
        self.result_path = os.path.join(self.output_path, model_name + "_" + self.dataset_name + ".xlsx")
        self.prev_file = os.path.join(self.output_path, model_name + "_" + self.dataset_name + "_PREV.json")

        # 保存每个卡上的结果
        self.out_file = self._out_file(self.rank)
        self.report_file = self.result_path.replace('.xlsx', '_acc.csv')
        is_divisive = len(dataset) % self.world_size == 0
        remainder = len(dataset) % self.world_size
        if is_divisive:
            data_len_total = len(dataset)
        elif drop_last and not is_divisive:
            raise ValueError("drop_last must be false now")
        elif not drop_last and not is_divisive:
            print(f"the length of dataset: {len(dataset)}, world_size: {self.world_size}, remainder: {remainder}")
            raise ValueError(f'The length of the dataset must be divided evenly by the world_size.')
        sheet_indices = list(range(self.rank, data_len_total, self.world_size))  # 对数据进行分块，后面涉及到多卡评估
        print(f"Rank {self.rank} of the data parallel group has {len(sheet_indices)} evaluation data ")
        self.data = dataset.data.iloc[sheet_indices]
        self.data_indices = [i for i in self.data['index']]  # 每个卡处理自己的index
        self.result = load_json(self.prev_file) if os.path.exists(self.prev_file) else {}

    def __call__(self):

        """
        调用pipeline开始进行评估，将评估结果写入文件中，
        """

        data = self.data[~self.data['index'].isin(self.result)]
        data_len = len(data)
        for i in tqdm(range(data_len)):
            data_index = data.iloc[i]['index']
            if data_index in self.result:
                # If data_index is already in result, skip it
                continue

            if self.model_prompt_template and self.model_prompt_template.is_use_custom_prompt(self.dataset_name):
                prompt = self.model_prompt_template.build_prompt(data.iloc[i], self.dataset.dump_image,
                                                                 self.dataset_name)
            else:
                prompt = self.dataset.build_prompt(data.iloc[i])

            BasePromptTemplate.check_content_type(prompt)
            instruct = BasePromptTemplate.preprocess_content(prompt)
            if instruct is None or BasePromptTemplate.check_content_type(instruct) != 'ListOfDict':
                raise ValueError(f'Invalid instruct: {instruct}. Only list of dict (ListOfDict) is supported.')
            for item in instruct:
                if item['type'] not in self.supported_types:
                    raise ValueError(
                        f'The instruct type is {item["type"]},only support {", ".join(self.supported_types)}')
            # To wait for all processes to finish processing the data
            dist.barrier()
            if hasattr(self.inference_pipeline, "evaluate"):
                response = self.inference_pipeline.evaluate(instruct)
            else:
                text, images = '', []
                for item in instruct:
                    if item['type'] == 'text':
                        text += item['value']
                    elif item['type'] == 'image':
                        images.append(item['value'])
                    else:
                        raise Exception("unsupported instruct type")
                response = self.inference_pipeline(prompt=text, images=images, return_ids=True)
            torch.cuda.empty_cache()
            if response is not None:
                print(response, flush=True)
                self.result[data_index] = response

            # 每20步存一下pkl文件
            if (
                ((i + 1) % 20 == 0)
                and mpu.is_pipeline_last_stage()
                and (mpu.get_tensor_model_parallel_rank() == 0)
            ):
                save_json(self.result, self.out_file)

        if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            res = {k: self.result[k] for k in self.data_indices}
            save_json(res, self.out_file)

    def gather_result(self):
        """
        获取每张卡上的计算结果

        """
        if self.world_size > 0:
            print("rank:", self.rank, " finish evaluate")
            dist.barrier()
        if torch.distributed.get_rank() == 0:
            data_all = {}
            for i in range(self.world_size):
                data_all.update(load_json(self._out_file(i)))

            data = self.dataset.data
            for x in data['index']:
                if x not in data_all:
                    raise ValueError(f"{x} not found in data_all")
            data['prediction'] = [str(data_all.get(x, None)) for x in data['index']]
            if 'image' in data:
                data.pop('image')

            # save to xlsx
            data.to_excel(self.result_path, index=False, engine='xlsxwriter')

            for i in range(self.world_size):
                os.remove(self._out_file(i))
        if self.world_size > 0:
            dist.barrier()

    def analyse_result(self):
        if self.world_size > 0:
            dist.barrier()
        if torch.distributed.get_rank() == 0:
            data = pd.read_excel(self.result_path)
            data = data.sort_values(by='index')
            data['prediction'] = [str(x) for x in data['prediction']]

            for k in data.keys():
                data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

            meta = self.dataset.data
            meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
            data_map = {x: y for x, y in zip(data['index'], data['question'])}

            for k in data_map:
                if k not in meta_q_map:
                    raise ValueError(
                        f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
                    )

            result = {}
            answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}

            if 'MMMU' in self.dataset_name:
                data = mmmu_open_question_preprocess(data)
                answer_map = {k: (v if v in list(string.ascii_uppercase) else 'A') for k, v in answer_map.items()}

            data = data[data['index'].isin(answer_map)]
            data['GT'] = [answer_map[idx] for idx in data['index']]
            items = []

            for i in range(len(data)):
                # Dealing with the normal part
                item = data.iloc[i]
                if item['index'] not in result:
                    items.append(item)

            tups = [dict(item=x, dataset_name=self.dataset_name) for x in items]
            keys = [x['index'] for x in items]

            nproc = 1

            if tups:
                res = track_progress_rich(eval_vanilla, tups, nproc=nproc, chunksize=nproc, save=self.prev_file,
                                          keys=keys)
                result = load_json(self.prev_file)
                for k, v in zip(keys, res):
                    result[k] = v
            data['hit'] = [result.get(i, {}).get('hit', None) for i in data['index']]
            data['log'] = [result.get(i, {}).get('log', None) for i in data['index']]
            if 'GT' in data:
                data.pop('GT')
            data.to_excel(self.result_path, index=False, engine='xlsxwriter')
            acc = report_acc(data)

            save_csv(acc, self.report_file)
            logger_rank_0(f"save acc file to {self.report_file}")
        if self.world_size > 0:
            dist.barrier()

    def _out_file(self, rank):
        return os.path.join(self.output_path,
                            str(rank) + "_" + f'{self.world_size}_{self.dataset_name}.json')
