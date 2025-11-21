# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Dict, Tuple, Union

import torch
import numpy

from megatron.core import InferenceParams, mpu
from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from megatron.training import get_args, print_rank_0

from mindspeed_mm.models.vlm_model import VLMModel
from mindspeed_mm.models.common.module_spec.get_layer_spec import get_vit_layer_spec, get_projector_layer_spec
from mindspeed_mm.models.vision.vision_model import Qwen3vlVisionModel
from mindspeed_mm.models.vision.vlm_attentionmask_for_llm import prepare_positionsids_mask_for_llm
from mindspeed_mm.utils.hetero_parallel import change_parallel_state
from mindspeed_mm.utils.utils import EncoderBalanceComm


class Qwen3VLModel(VLMModel):
    """
    Vision-Language multi-modal model.
    VLMModel is an assembled model, which include image_encoder, text_decoder model.

    Args:
        config (dict): the general config for VLModel, model.json中的配置
        {
            "pre_process": (bool),  # Include the embedding leayer in the gpt decoder (used with pipeline parallelism).
            "post_process": (bool),  # Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism).
            "add_text_encoder": (bool),  # Whether to construct the text encoder. not used now.
            "reward_process: (bool, optional), # Without an output layer in the gpt decoder (only used with videoalign). Defaults to False.
            "add_text_encoder": (bool),  # Whether to construct the text encoder. not used now.
            "add_image_encoder": (bool),  # Whether to construct the image encoder.
            "add_video_encoder": (bool),  # Whether to construct the video encoder. not used now.
            "add_text_decoder": (bool),  # Whether to construct the text decoder.
            "img_context_token_id": (int),  # Index in the language_embeddings tensor where image_embeddings should be inserted.
            "text_encoder": {...},  # Config for the text encoder. not used now.
            "image_encoder": {...},  # Config for the image encoder.
            "video_encoder": {...},  # Config for the video encoder. not used now.
            "text_decoder": {...},  # Config for the text decoder.
        }
    """

    def _build_image_encoder_model(self, config):
        self.encoder_dp_enable = config.vision_encoder.model_id == "InternViT"

        if get_args().hetero_parallel:
            change_parallel_state('image_encoder')

            self.pp_size = mpu.get_pipeline_model_parallel_world_size()
            self.enable_vp = mpu.get_virtual_pipeline_model_parallel_world_size() is not None
            if self.enable_vp:
                self.vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
                self.vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()
            print_rank_0(f'initial: image_encoder pp size is {self.pp_size}')
            print_rank_0(f'initial: image_encoder tp size is {mpu.get_tensor_model_parallel_world_size()}')
            print_rank_0(f'initial: image_encoder cp size is {mpu.get_context_parallel_world_size()}')
            print_rank_0(f'initial: image_encoder dp size is {mpu.get_data_parallel_world_size()}')

        vit_layer_spec = get_vit_layer_spec(config.vision_encoder)
        proj_layer_spec = get_projector_layer_spec(config.vision_projector)

        if self.pp_size <= 1:
            return Qwen3vlVisionModel(
                config=config,
                encoder_transformer_layer_spec=vit_layer_spec,
                projector_layer_spec=proj_layer_spec
            )
        if self.enable_vp:
            if self.pp_size * self.vp_size != len(config.vision_encoder.pipeline_num_layers) * len(
                    config.vision_encoder.pipeline_num_layers[0]):
                raise ValueError(
                    f"The product of pipeline-model-parallel-size and vpp-size must equal to the total number of stage in vision_encoder.pipeline_num_layers, "
                    f"but got pipeline-model-parallel-size: {self.pp_size}, vpp-size: {self.vp_size}, "
                    f"and total number of stage in vision_encoder.pipeline_num_layers: {len(config.vision_encoder.pipeline_num_layers) * len(config.vision_encoder.pipeline_num_layers[0])}.")
        elif self.pp_size != len(config.vision_encoder.pipeline_num_layers):
            raise ValueError(
                f"length of vision_encoder.pipeline_num_layers must equal to pipeline-model-parallel-size, "
                f"but got vision_encoder.pipeline_num_layers length:{len(config.vision_encoder.pipeline_num_layers)} "
                f"and pipeline-model-parallel-size:{self.pp_size}.")

        if self.enable_vp:
            local_num_layers = config.vision_encoder.pipeline_num_layers[self.vp_rank][self.pp_rank]
        else:
            local_num_layers = config.vision_encoder.pipeline_num_layers[self.pp_rank]

        if local_num_layers == 0:
            self.add_image_encoder = False
            return None

        if self.enable_vp:
            pipeline_start_index = sum(
                sum(vp_layer) for vp_layer in config.vision_encoder.pipeline_num_layers[:self.vp_rank]) + sum(
                config.vision_encoder.pipeline_num_layers[self.vp_rank][:self.pp_rank])
            pipeline_end_index = sum(
                sum(vp_layer) for vp_layer in config.vision_encoder.pipeline_num_layers[:self.vp_rank]) + sum(
                config.vision_encoder.pipeline_num_layers[self.vp_rank][:self.pp_rank + 1])
        else:
            pipeline_start_index = sum(config.vision_encoder.pipeline_num_layers[:self.pp_rank])
            pipeline_end_index = sum(config.vision_encoder.pipeline_num_layers[:self.pp_rank + 1])

        pre_process = pipeline_start_index == 0
        post_process = pipeline_end_index == config.vision_encoder.num_layers

        print(
            f"image encoder pipeline config:\
            pp_rank:{self.pp_rank},\
            pre_process:{pre_process},\
            post_process:{post_process},\
            local_num_layers:{local_num_layers}"
        )
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        config.vision_encoder.num_layers = self.pp_size * local_num_layers
        if self.enable_vp:
            config.vision_encoder.num_layers *= self.vp_size
        return Qwen3vlVisionModel(
            config=config,
            encoder_transformer_layer_spec=vit_layer_spec,
            projector_layer_spec=proj_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
        )

    def custom_scatter(self, input_tensor):
        # layout of input tensor should be SBH
        S, B, H = input_tensor.shape
        output_tensor = input_tensor
        if get_args().tensor_model_parallel_size > 1:
            input_tensor = input_tensor.reshape(-1, H).contiguous()
            output_tensor = tensor_parallel.scatter_to_sequence_parallel_region(input_tensor)
            output_tensor = output_tensor.reshape(-1, B, H).contiguous()

        return output_tensor

    def forward(
            self,
            input_ids: torch.Tensor,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            inference_params: Optional[InferenceParams] = None,
            decoder_input: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            packed_seq_params: Optional[PackedSeqParams] = None,
            extra_block_kwargs: Optional[dict] = None,
            cache_position: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            transfer: Optional[numpy.ndarray] = None,
            *args, **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        deepstack_image_embeds = None

        # MM_GRPO use, if llm_only is True, directly get vit_embeds
        if self.add_image_encoder and self.image_encoder.pre_process and kwargs.get('llm_only', False):
            vit_embeds = kwargs.get('vit_embeds').unsqueeze(1)
        elif self.add_image_encoder and pixel_values is not None:
            vit_embeds, deepstack_image_embeds = self.image_encoder(pixel_values, image_grid_thw)
            if get_args().encoder_dp_balance and self.encoder_dp_enable:
                vit_embeds = EncoderBalanceComm.apply(
                    vit_embeds,
                    mpu.get_data_parallel_group(),
                    transfer
                )

            if image_flags is not None:
                if self.image_encoder.post_process:
                    image_flags = image_flags.squeeze(-1)
                    vit_embeds = vit_embeds[image_flags == 1]
                    vit_embeds = vit_embeds.reshape(-1, 1, vit_embeds.shape[-1]).clone()
            else:
                vit_embeds = vit_embeds.reshape(-1, 1, vit_embeds.shape[-1]).clone()
            output = vit_embeds
        else:
            vit_embeds = self.input_tensor

        # MM_GRPO use, if vit_only is True, only calculate vit_embeds and return
        if kwargs.get('vit_only', False) and self.image_encoder.post_process:
            return {"vit_embeds": vit_embeds}

        if self.add_text_decoder:
            deepstack_visual_embeds = None
            input_embeds = None
            if self.text_decoder.pre_process:
                input_embeds = self.text_decoder.embedding(input_ids=input_ids, position_ids=position_ids).clone()
                
                if self.config.sequence_parallel:
                    input_embeds = gather_from_sequence_parallel_region(input_embeds, tensor_parallel_output_grad=False)

                if vit_embeds is not None:
                    input_embeds = input_embeds.transpose(0, 1)  # bsh
                    image_mask = torch.eq(input_ids, self.img_context_token_id)
                    vit_embeds = vit_embeds[:, 0, :]
                    indices_tuple = torch.nonzero(image_mask, as_tuple=True)
                    input_embeds[indices_tuple] = vit_embeds

                    deepstack_visual_embeds = []
                    if deepstack_image_embeds:
                        for deepstack_image in deepstack_image_embeds:
                            if self.config.sequence_parallel:
                                deepstack_image = gather_from_sequence_parallel_region(deepstack_image, tensor_parallel_output_grad=False)
                                deepstack_image = deepstack_image[: vit_embeds.shape[0], :]

                            deepstack_emb = deepstack_image.new_zeros(input_embeds.shape)
                            deepstack_emb[indices_tuple] = deepstack_image

                            deepstack_emb = deepstack_emb.transpose(0, 1)
                            deepstack_emb = self.custom_scatter(deepstack_emb)
                            deepstack_visual_embeds.append(deepstack_emb)
                    input_embeds = input_embeds.transpose(0, 1)


            attention_mask, position_ids = prepare_positionsids_mask_for_llm(config=self.config, input_ids=input_ids,
                                                                             inference_params=inference_params,
                                                                             attention_mask=attention_mask,
                                                                             position_ids=position_ids,
                                                                             image_grid_thw=image_grid_thw,
                                                                             rope_deltas=rope_deltas,
                                                                             inputs_embeds=input_embeds,
                                                                             cache_position=cache_position,
                                                                             **kwargs)

            output = self.text_decoder(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=input_embeds,
                labels=None,
                inference_params=inference_params,
                extra_block_kwargs={'deepstack_visual_embeds': deepstack_visual_embeds}
            )

            if self.text_decoder.post_process:
                output = output.contiguous().float()
                loss_dict = {}
                if labels is not None:
                    if mpu.get_context_parallel_world_size() > 1:
                        loss, token_nums = self.compute_loss_with_context_parallel(output, labels)

                        loss_dict["loss"] = loss
                        loss_dict["token_nums"] = token_nums

                        return {
                            "loss_dict": loss_dict,
                            "logits": output
                        }
                    else:
                        # output shape [b, s, vocab_size]
                        shift_logits = output[..., :-1, :].contiguous()
                        # labels shape [b, s]
                        shift_labels = labels[..., 1:].contiguous()

                        if mpu.get_tensor_model_parallel_world_size() > 1:
                            loss = self.compute_loss_with_tensor_parallel(shift_logits, shift_labels)
                        else:
                            loss = self.compute_language_model_loss(shift_logits, shift_labels)

                        loss_dict["loss"] = loss
                        loss_dict["loss_mask"] = shift_labels > -1

                        return {
                            "loss_dict": loss_dict,
                            "logits": output
                        }

                return {
                    "loss": None,
                    "logits": output
                }
            
        return output
