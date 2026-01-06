# Copyright (c) OpenMMLab. All rights reserved.
# Partly adopted from https://github.com/GT-Vision-Lab/VQA
# Copyright (c) 2014, Aishwarya Agrawal

import copy
import os
import re
import string
import ast

import pandas as pd
import torch
from torch import distributed as dist


def dict2dataframe(d):
    return pd.DataFrame({x: [d[x]] for x in d})


def string_to_list(s):
    if isinstance(s, str) and s.startswith('[') and s.endswith(']'):
        return [str(x) for x in ast.literal_eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError


def is_expected_type(s, expected_type):
    if isinstance(s, expected_type):
        return True
    try:
        return isinstance(ast.literal_eval(s), expected_type)
    except Exception as _:
        return False


def is_cn_string(s):
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False


def build_option_str(option_dict):
    s = 'There are several options: \n'
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s


def extract_choices(item):
    """
    Extract choices from a question.
    Args:
        item: A question, typically a dictionary containing options.
    Returns:
        A dictionary containing valid options.
    """
    ret = {}
    for ch in string.ascii_uppercase:
        if ch in item and (not pd.isna(item[ch])):
            ret[ch] = item[ch]
    return ret


def check_answer_in_text(answer, choices):
    answer = answer.lower()
    if not isinstance(choices, dict):
        raise TypeError("Expected 'choices' to be a dict.")
    for k, v in choices.items():
        if k not in string.ascii_uppercase:
            raise ValueError(f"Expected 'k' to be an uppercase letter, got '{k}'.")
        choices[k] = str(v).lower()
    cands = [k for k, v in choices.items() if v in answer]
    return cands[0] if len(cands) == 1 else False


def extract_answer_from_item(item, dataset_name=None):
    logger_rank_0('Evaluation')
    # It will return: (pred, raw, llm_time)
    choices = extract_choices(item)  # extract choices

    ret = check_answer(item['prediction'], choices)
    if ret:
        return dict(opt=ret, log=item['prediction'])
    else:
        return dict(opt='Z', log='Failed in Prefetch, no GPT-based answer matching under `exact_matching` policy.')


def check_answer_in_option(answer, choices):
    answer = str(answer)
    verbose = os.environ.get('VERBOSE', 0)
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]

    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    matches = [ch for ch in choices if ch in splits]

    if len(matches) == 1:
        match = matches.pop()
        if 'A' in splits and len(splits) > 3 and verbose:
            logger_rank_0(f'A might be a quantifier in the string: {answer}.')
            return False
        return match
    elif not matches and ('Z' in splits or '' in splits):
        return 'Z'
    return False


def check_answer(answer, choices):
    if not isinstance(answer, str):
        answer = str(answer)
    option_result = check_answer_in_option(answer, choices)
    if not option_result:
        return check_answer_in_text(answer, choices)
    else:
        return option_result


def logger_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_world_size_and_rank():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        raise Exception('torch distributed is not initialized')
    return rank, world_size


def process_punctuation(in_text):
    out_text = in_text
    punctuation = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    comma_strip = re.compile('(\d)(,)(\d)')
    period_strip = re.compile('(?!<=\d)(\.)(?!\d)')
    for p in punctuation:
        if (p + ' ' in in_text or ' ' + p in in_text) or (re.search(
                comma_strip, in_text) is not None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = period_strip.sub('', out_text, re.UNICODE)
    return out_text


def _process_digit_article(in_text):
    out_text = []
    temp_text = in_text.lower().split()
    articles = ['a', 'an', 'the']
    manual_map = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    contractions = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }
    for word in temp_text:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            out_text.append(word)
    for wordId, word in enumerate(out_text):
        if word in contractions:
            out_text[wordId] = contractions[word]
    out_text = ' '.join(out_text)
    return out_text


def is_list_in_str(list_input, s):
    if not isinstance(list_input, list):
        raise TypeError(f"Expected 'list_input' to be of type list, but got {type(list_input).__name__}.")
    for item in list_input:
        if item.lower() in s:
            return True
    return False


def process_answer(answer):
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\t', ' ')
    answer = answer.strip()
    answer = process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer


