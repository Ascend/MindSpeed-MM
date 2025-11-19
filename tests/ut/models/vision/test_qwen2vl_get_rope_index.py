from unittest.mock import MagicMock
import mindspeed.megatron_adaptor
import pytest
import torch
from tests.ut.utils import judge_expression
from mindspeed_mm.models.vision.vlm_attentionmask_for_llm import qwen2vl_get_rope_index



@pytest.fixture
def mock_args():
    args = MagicMock()
    args.mm.model.vision_start_token_id = 151652
    args.mm.model.image_token_id = 151655
    args.mm.model.video_token_id = 151656
    args.mm.model.image_encoder.vision_encoder.spatial_merge_size = 2
    args.mm.model.image_encoder.vision_encoder.tokens_per_second = 2
    return args


@pytest.fixture(autouse=True)
def patch_get_args(mock_args, mocker):
    mocker.patch("mindspeed_mm.models.vision.vlm_attentionmask_for_llm.get_args", return_value=mock_args)


class TestQwen2VLGetRoPEIndex:

    def test_pure_text_input_with_attention_mask(self):
        input_ids = torch.tensor([
            [151644, 101, 102, 103],
            [151644, 104, 105, 151643],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ])

        pos_ids, deltas = qwen2vl_get_rope_index(
            config=None,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        expected_deltas = torch.tensor([[0], [-1]])
        judge_expression(torch.equal(deltas, expected_deltas))

        expected_pos_ids = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 1]],
                                         [[0, 1, 2, 3], [0, 1, 2, 1]],
                                         [[0, 1, 2, 3], [0, 1, 2, 1]]])
        judge_expression(torch.equal(pos_ids, expected_pos_ids))


    def test_pure_text_input_without_attention_mask(self):
        input_ids = torch.tensor([[151644, 101, 102, 103]])
        pos_ids, deltas = qwen2vl_get_rope_index(
            config=None,
            input_ids=input_ids
        )
        
        expected_deltas = torch.tensor([[0]])
        judge_expression(torch.equal(deltas, expected_deltas))

        expected_pos_ids = torch.tensor([[[0, 1, 2, 3]],
                                         [[0, 1, 2, 3]],
                                         [[0, 1, 2, 3]]])
        judge_expression(torch.equal(pos_ids, expected_pos_ids))


    def test_video_and_text_input(self):
        input_ids = torch.tensor([[151644, 101, 102, 103, 151652, 151656, 151656, 151656, 151656, 151656, 151656, 151656, 151656,
                                   151656, 151656, 151656, 151656, 151653, 151644, 104, 105, 106]])
        image_grid_thw = torch.tensor([[2, 4, 6]])
        pos_ids, deltas = qwen2vl_get_rope_index(
            config=None,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw
        )

        expected_deltas = torch.tensor([[-9]])
        judge_expression(torch.equal(deltas, expected_deltas))

        expected_pos_ids = torch.tensor([[[0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 8, 9, 10, 11, 12]],
                                         [[0, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 5, 5, 5, 6, 6, 6, 8, 9, 10, 11, 12]],
                                         [[0, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 12]]])
        judge_expression(torch.equal(pos_ids, expected_pos_ids))