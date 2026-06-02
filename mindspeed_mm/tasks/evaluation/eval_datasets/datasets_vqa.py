from mindspeed_mm.tasks.evaluation.eval_datasets.datasets_base import BaseEvalDataset


class VQAEvalDataset(BaseEvalDataset):

    def __init__(self, text_path, image_path):
        super().__init__(text_path, image_path)

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        if msgs[-1]['type'] != 'text':
            raise ValueError(
                "The last message in the list is expected to be of type 'text', but got '{}'.".format(msgs[-1]['type']))
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
        return msgs
