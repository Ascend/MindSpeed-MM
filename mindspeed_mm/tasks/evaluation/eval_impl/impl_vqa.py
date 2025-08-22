import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import torch.distributed as dist

from mindspeed_mm.tasks.evaluation.eval_impl.impl_base import BaseEvalImpl
from mindspeed_mm.tasks.evaluation.utils.analysis_utils import hit_calculate, process_line
from mindspeed_mm.tasks.evaluation.utils.string_utils import is_list_in_str, dict2dataframe, logger_rank_0
from mindspeed_mm.utils.security_utils.input_filter import sanitize_dataframe


class VQAEvalImpl(BaseEvalImpl):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self):
        super().__call__()
        self.gather_result()
        self.analyse_result()

    def analyse_result(self):
        if self.world_size > 0:
            dist.barrier()
        if dist.get_rank() == 0:
            data = pd.read_excel(self.result_path)
            dataset = self.dataset_name
            if 'answer' not in data or 'prediction' not in data:
                raise ValueError("Data must contain both 'answer' and 'prediction' keys.")
            data['prediction'] = [str(x) for x in data['prediction']]
            data['answer'] = [str(x) for x in data['answer']]
            lt = len(data)
            pool = mp.Pool(1)
            lines = [data.iloc[i] for i in range(lt)]
            if is_list_in_str(['chartqa'], dataset):
                res = pool.map(partial(process_line, method='relaxed_accuracy'), lines)
            elif is_list_in_str(['docvqa', 'infovqa'], dataset):
                res = pool.map(partial(process_line, method='anls'), lines)
            else:  # default using vqa_score to calculate score
                res = pool.map(process_line, lines)
            pool.close()
            hit = hit_calculate(res, dataset)
            ret = dict()
            if 'split' in data:
                splits = set(data['split'])
                for sp in splits:
                    sub = [result for line, result in zip(lines, res) if line['split'] == sp]
                    hit = hit_calculate(sub, dataset)
                    ret[sp] = np.mean(hit) * 100
                sub = [result for line, result in zip(lines, res)]
                hit = hit_calculate(sub, dataset)
                ret['Overall'] = np.mean(hit) * 100
            else:
                ret['Overall'] = np.mean(hit) * 100
                if 'category' in data:
                    cates = list(set(data['category']))
                    cates.sort()
                    for c in cates:
                        sub = [result for line, result in zip(lines, res) if line['category'] == c]
                        hit = hit_calculate(sub, dataset)
                        ret[c] = np.mean(hit) * 100
            ret = dict2dataframe(ret)
            ret.round(2)

            suffix = self.result_path.split('.')[-1]
            result_file = self.result_path.replace(f'.{suffix}', '_acc.xlsx')

            cleaned_ret = sanitize_dataframe(ret)
            cleaned_ret.to_excel(result_file, index=False, engine='xlsxwriter')
            logger_rank_0(f"save acc file to {result_file}")
        if self.world_size > 0:
            dist.barrier()
