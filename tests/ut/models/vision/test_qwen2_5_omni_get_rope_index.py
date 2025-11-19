from unittest.mock import MagicMock
import mindspeed.megatron_adaptor
import pytest
import torch
from tests.ut.utils import judge_expression
from mindspeed_mm.models.vision.vlm_attentionmask_for_llm import qwen2_5_omni_get_rope_index



@pytest.fixture
def mock_args():
    args = MagicMock()
    args.mm.model.vision_start_token_id = 151652
    args.mm.model.image_token_id = 151655
    args.mm.model.video_token_id = 151656
    args.mm.model.audio_start_token_id = 151647
    args.mm.model.image_encoder.vision_encoder.spatial_merge_size = 2
    args.mm.model.image_encoder.vision_encoder.tokens_per_second = 2
    return args


@pytest.fixture(autouse=True)
def patch_get_args(mock_args, mocker):
    mocker.patch("mindspeed_mm.models.vision.vlm_attentionmask_for_llm.get_args", return_value=mock_args)


class TestQwen2_5OminiGetRoPEIndex:

    def test_pure_text_input(self):
        input_ids = torch.tensor([
            [151644, 101, 102, 103],
            [151644, 104, 105, 151643],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ])

        pos_ids, deltas = qwen2_5_omni_get_rope_index(
            config=None,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        expected_deltas = torch.tensor([[0], [0]])
        judge_expression(torch.equal(deltas, expected_deltas))

        expected_pos_ids = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 1]],
                                         [[0, 1, 2, 3], [0, 1, 2, 1]],
                                         [[0, 1, 2, 3], [0, 1, 2, 1]]])
        judge_expression(torch.equal(pos_ids, expected_pos_ids))


    def test_image_and_text_input(self):
        image_grid_thw = torch.tensor([[1, 4, 6]])
        input_ids = torch.tensor([[151644, 101, 102, 151652, 151655, 151655, 151655, 151655, 151655, 151655, 151653, 103, 104]])

        pos_ids, deltas = qwen2_5_omni_get_rope_index(
            config=None,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw
        )

        judge_expression(deltas == torch.tensor([[-3]]))
        expected_pos_ids = torch.tensor([[[0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 7, 8, 9]],
                                         [[0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 7, 8, 9]],
                                         [[0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9]]])
        
        judge_expression(torch.equal(pos_ids, expected_pos_ids))

    
    def test_use_audio_in_video_input(self):
        image_grid_thw = torch.tensor([[2, 4, 6]])
        second_per_grid = torch.tensor([1.0])
        audio_seqlens = torch.tensor([48])
        input_ids = torch.tensor([[151644, 101, 102, 151652, 151647, 151656, 151656, 151656, 151656, 151656, 151656, 151656, 151656,
                                   151656, 151656, 151656, 151656, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 
                                   151646, 151646, 151646, 151646, 151648, 151653, 151644, 103, 104]])
        attention_mask = torch.ones_like(input_ids)

        pos_ids, deltas = qwen2_5_omni_get_rope_index(
            config=None,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            second_per_grid=second_per_grid,
            audio_seqlens=audio_seqlens,
            use_audio_in_video=True,
            attention_mask=attention_mask
        )

        judge_expression(deltas == torch.tensor([[-4]]))
        expected_pos_ids = torch.tensor([[[0, 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 29, 29, 29, 29, 29, 29, 4, 
                                           5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18, 19]],
                                         [[0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4,  
                                           5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18, 19]],
                                         [[0, 1, 2, 3, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4,  
                                           5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18, 19]]])
        judge_expression(torch.equal(pos_ids, expected_pos_ids))


    def test_not_use_audio_in_video_input(self):
        image_grid_thw = torch.tensor([[2, 4, 6]])
        second_per_grid = torch.tensor([1.0])
        audio_seqlens = torch.tensor([48])
        input_ids = torch.tensor([[151644, 101, 102, 151652, 151656, 151656, 151656, 151656, 151656, 151656, 151656, 151656,
                                   151656, 151656, 151656, 151656, 151653, 151644, 103, 104, 151647, 151646, 151646, 151646, 
                                   151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151648]])
        attention_mask = torch.ones_like(input_ids)

        pos_ids, deltas = qwen2_5_omni_get_rope_index(
            config=None,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            second_per_grid=second_per_grid,
            audio_seqlens=audio_seqlens,
            use_audio_in_video=False,
            attention_mask=attention_mask
        )

        judge_expression(deltas == torch.tensor([[14]]))
        expected_pos_ids = torch.tensor([[[0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 29, 29, 29, 29, 29, 29, 30, 31, 
                                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]],
                                         [[0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 5, 5, 30, 31, 
                                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]],
                                         [[0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 30, 31, 
                                           32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]]])
        judge_expression(torch.equal(pos_ids, expected_pos_ids))