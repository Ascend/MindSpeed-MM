import ast
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import Callable, Iterable, Sized, Optional

import numpy as np
import pandas as pd
import portalocker
from rich.progress import (Progress, TextColumn, BarColumn, MofNCompleteColumn, TaskProgressColumn,
                           TimeRemainingColumn, Task)
from rich.text import Text

from mindspeed_mm.tasks.evaluation.utils.file_utils import save_json, load_json
from mindspeed_mm.tasks.evaluation.utils.string_utils import extract_answer_from_item, logger_rank_0, is_expected_type, \
    process_answer
from mindspeed_mm.tasks.evaluation.utils.string_utils import is_list_in_str


def mmmu_open_question_preprocess(data):
    # 使用向量化操作填充 A选项的空值, 如果A选项为空则为开放性问答
    mask = data['A'].isna()
    data.loc[mask, 'A'] = data.loc[mask, 'answer']
    data.loc[mask, 'B'] = 'Other Answers'
    # 记录开放性问题数量
    cnt = mask.sum()
    logger_rank_0(f'During MMMU_preproc in Evaluation, {cnt} open questions are re-formulated to multi-choice ones.')
    return data


def report_acc(data: pd.DataFrame):
    res = defaultdict(list)
    if 'split' not in data:
        data['split'] = 'none'
    splits = data['split'].unique()
    res['split'] = splits

    def calculate_accuracy(sub_df, splits):
        # 按照splits类型划分不同的分数
        return [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in splits]

    res['Overall'] = calculate_accuracy(data, splits)
    for group in ['l2-category', 'category']:
        if group in data:
            abilities = sorted(set(data[group]))
            # 按照里面的category类型 统计分数
            for ab in abilities:
                ab_name = MMB_abbrs[ab] if ab in MMB_abbrs else ab
                sub_df = data[data[group] == ab]
                res[ab_name] = calculate_accuracy(sub_df, splits)
    return pd.DataFrame(res)


def eval_vanilla(item, dataset_name=None):
    res = extract_answer_from_item(item, dataset_name=dataset_name)
    opt, match_log = res['opt'], res['log']
    if opt == item['GT']:
        return dict(hit=1, log=f'Match Log: {match_log}. ')
    else:
        return dict(hit=0, log=f'Match Log: {match_log}. ')


MMB_abbrs = {
    'coarse_perception': 'CP',
    'finegrained_perception (instance-level)': 'FP-S',
    'finegrained_perception (cross-instance)': 'FP-C',
    'logic_reasoning': 'LR',
    'relation_reasoning': 'RR',
    'attribute_reasoning': 'AR'
}


def track_progress_rich(func: Callable,
                        tasks: Iterable = tuple(),
                        task_num: int = None,
                        nproc: int = 1,
                        chunksize: int = 1,
                        description: str = 'Processing',
                        save=None, keys=None,
                        color: str = 'blue') -> list:
    """Track the progress of parallel task execution with a progress bar. The
    built-in :mod:`multiprocessing` module is used for process pools and tasks
    are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (Iterable or Sized): A tuple of tasks. There are several cases
            for different format tasks:
            - When ``func`` accepts no arguments: tasks should be an empty
              tuple, and ``task_num`` must be specified.
            - When ``func`` accepts only one argument: tasks should be a tuple
              containing the argument.
            - When ``func`` accepts multiple arguments: tasks should be a
              tuple, with each element representing a set of arguments.
              If an element is a ``dict``, it will be parsed as a set of
              keyword-only arguments.
            Defaults to an empty tuple.
        task_num (int, optional): If ``tasks`` is an iterator which does not
            have length, the number of tasks can be provided by ``task_num``.
            Defaults to None.
        nproc (int): Process (worker) number, if nuproc is 1,
            use single process. Defaults to 1.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
            Defaults to 1.
        description (str): The description of progress bar.
            Defaults to "Process".
        color (str): The color of progress bar. Defaults to "blue".

    Examples:
        >>> import time

        >>> def func(x):
        ...    time.sleep(1)
        ...    return x**2
        >>> track_progress_rich(func, range(10), nproc=2)

    Returns:
        list: The task results.
    """
    if save is not None:
        if not (os.path.exists(os.path.dirname(save)) or os.path.dirname(save) == ''):
            raise FileNotFoundError(f"The directory part of the path '{save}' does not exist.")

        if not os.path.exists(save):
            save_json({}, save)
    if keys is not None:
        if len(keys) != len(tasks):
            raise ValueError(f"The length of 'keys' ({len(keys)}) "
                             f"does not match the length of 'tasks' ({len(tasks)}).")

    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    if isinstance(tasks, Sized):
        if len(tasks) == 0:
            if task_num is None:
                raise ValueError('If tasks is an empty iterable, '
                                 'task_num must be set')
            else:
                tasks = tuple(tuple() for _ in range(task_num))
        else:
            if task_num is not None and task_num != len(tasks):
                raise ValueError('task_num does not match the length of tasks')
            task_num = len(tasks)

    if nproc <= 0:
        raise ValueError('nproc must be a positive number')

    skip_times = nproc * chunksize if nproc > 1 else 0
    prog_bar = Progress(
        TextColumn('{task.description}'),
        BarColumn(),
        _SkipFirstTimeRemainingColumn(skip_times=skip_times),
        MofNCompleteColumn(),
        TaskProgressColumn(show_speed=True),
    )

    worker = _Worker(func)
    task_id = prog_bar.add_task(
        total=task_num, color=color, description=description)
    tasks = _tasks_with_index(tasks)

    # Use single process when nproc is 1, else use multiprocess.
    with prog_bar:
        if nproc == 1:
            results = []
            for task in tasks:
                result, idx = worker(task)
                results.append(result)
                if save is not None:
                    with portalocker.Lock(save, timeout=5) as fh:
                        ans = load_json(save)
                        ans[keys[idx]] = result

                        if os.environ.get('VERBOSE', True):
                            print(keys[idx], result, flush=True)

                        save_json(ans, save)
                        fh.flush()
                        os.fsync(fh.fileno())

                prog_bar.update(task_id, advance=1, refresh=True)
        else:
            with Pool(nproc) as pool:
                results = []
                unordered_results = []
                gen = pool.imap_unordered(worker, tasks, chunksize)
                try:
                    for result in gen:
                        result, idx = result
                        unordered_results.append((result, idx))

                        if save is not None:
                            with portalocker.Lock(save, timeout=5) as fh:
                                ans = load_json(save)
                                ans[keys[idx]] = result

                                if os.environ.get('VERBOSE', False):
                                    print(keys[idx], result, flush=True)

                                save_json(ans, save)
                                fh.flush()
                                os.fsync(fh.fileno())

                        results.append(None)
                        prog_bar.update(task_id, advance=1, refresh=True)
                except Exception as e:
                    prog_bar.stop()
                    raise e
            for result, idx in unordered_results:
                results[idx] = result
    return results


class _SkipFirstTimeRemainingColumn(TimeRemainingColumn):
    """Skip calculating remaining time for the first few times.

    Args:
        skip_times (int): The number of times to skip. Defaults to 0.
    """

    def __init__(self, *args, skip_times=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_times = skip_times

    def render(self, task: Task) -> Text:
        """Show time remaining."""
        if task.completed <= self.skip_times:
            return Text('-:--:--', style='progress.remaining')
        return super().render(task)


class _Worker:
    """Function wrapper for ``track_progress_rich``"""

    def __init__(self, func) -> None:
        self.func = func

    def __call__(self, inputs):
        inputs, idx = inputs
        if not isinstance(inputs, (tuple, list, dict)):
            inputs = (inputs,)

        if isinstance(inputs, dict):
            return self.func(**inputs), idx
        else:
            return self.func(*inputs), idx


def _tasks_with_index(tasks):
    """Add index to tasks."""
    for idx, task in enumerate(tasks):
        yield task, idx


def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction = str(prediction)
    target = str(target)
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def anls_compute(groundtruth, prediction):
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    values = 0.0 if length == 0 else float(dist) / float(length)
    return values


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def hit_calculate(result, dataset_name, anls_threshold=0.5):
    if is_list_in_str(['textvqa'], dataset_name):
        return [np.mean(x['match']) for x in result]
    elif is_list_in_str(['docvqa', 'infovqa'], dataset_name):
        return [0.0 if 1 - np.min(x['match']) < anls_threshold else 1 - np.min(x['match']) for x in result]
    elif is_list_in_str(['chartqa', 'ocrvqa'], dataset_name):
        return [np.max(x['match']) for x in result]
    else:  # default using vqa_score to calculate score
        return [np.mean(x['match']) for x in result]


def process_line(line, method='vqa_score'):
    ret = {}
    if is_expected_type(line['answer'], list):
        answers = ast.literal_eval(line['answer'])
    else:
        answers = [line['answer']]
    if method == 'vqa_score':
        ret['gt'] = [process_answer(x) for x in answers]
        ret['pred'] = process_answer(line['prediction'])
        ret['match'] = []
        for current_idx, _ in enumerate(ret['gt']):
            other_gt_ans = [
                item
                for ret_gt_idx, item in enumerate(ret['gt'])
                if ret_gt_idx != current_idx
            ]
            matching_ans = [
                item
                for item in other_gt_ans
                if item == ret['pred']
            ]
            acc = min(1, float(len(matching_ans)) / 3)
            ret['match'].append(acc)
    elif method == 'anls':
        ret['gt'] = answers
        ret['pred'] = line['prediction']
        ret['match'] = [anls_compute(x, ret['pred']) for x in ret['gt']]
    elif method == 'relaxed_accuracy':
        ret['gt'] = answers
        ret['pred'] = line['prediction'].strip()
        ret['match'] = [relaxed_correctness(ret['pred'], x) for x in ret['gt']]
    elif method == 'accuracy':
        ret['gt'] = answers
        ret['pred'] = line['prediction'].strip()
        ret['match'] = [(1.0 if (x.strip().lower() == ret['pred'].strip().lower()) else 0.0) for x in ret['gt']]
    else:  # default using vqa_score to calculate score
        ret['gt'] = [process_answer(x) for x in answers]
        ret['pred'] = process_answer(line['prediction'])
        ret['match'] = [x == ret['pred'] for x in ret['gt']]

    return ret
