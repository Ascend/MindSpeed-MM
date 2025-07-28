# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Dict, Tuple, Union
import torch
from torch.nn import CrossEntropyLoss

from megatron.core import InferenceParams, mpu
from megatron.core import tensor_parallel
from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.module_spec.get_layer_spec import get_vit_layer_spec, get_llm_layer_spec, \
    get_projector_layer_spec, get_audio_layer_spec
from mindspeed_mm.models.vision.vision_model import VisionModel
from mindspeed_mm.models.audio.audio_model import AudioModel
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.text_encoder.text_encoder import TextEncoder
from mindspeed_mm.models.common.mm_gpt_model import MMGPTModel
from mindspeed_mm.models.vision.vlm_attentionmask_for_llm import prepare_positionsids_mask_for_llm


class VLMModel(MultiModalModule):
    """
    Vision-Language multi-modal model.
    VLMModel is an assembled model, which include image_encoder, text_decoder model.

    Args:
        config (dict): the general config for VLModel, model.json中的配置
        {
            "pre_process": (bool),  # Include the embedding leayer in the gpt decoder (used with pipeline parallelism).
            "post_process": (bool),  # Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism).
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

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.config = core_transformer_config_from_args(get_args())
        self.pre_process: bool = config.pre_process
        self.post_process: bool = config.post_process
        self.add_text_encoder = config.text_encoder is not None
        self.add_image_encoder = config.image_encoder is not None
        self.add_video_encoder = config.video_encoder is not None
        self.add_text_decoder = config.text_decoder is not None
        self.add_audio_encoder = hasattr(config, "audio_encoder") and config.audio_encoder is not None

        self.text_encoder = None
        self.image_encoder = None
        self.video_encoder = None
        self.text_decoder = None

        self.share_embeddings_and_output_weights = not getattr(config.text_decoder, 'untie_embeddings_and_output_weights', True)
        self.img_context_token_id = config.img_context_token_id
        

        # initialize pipeline parallel configs
        self.pp_size = mpu.get_pipeline_model_parallel_world_size()
        self.enable_vp = mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        if self.enable_vp:
            self.vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            self.vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        
        if self.add_text_encoder:
            self.text_encoder = TextEncoder(config.text_encoder).get_model()
        if self.add_image_encoder:
            self.image_encoder = self._build_image_encoder_model(config.image_encoder)
        if self.add_video_encoder:
            raise NotImplementedError("Not support video_encoder now")
        if self.add_text_decoder:
            self.position_embedding_type = config.text_decoder.position_embedding_type
            self.vocab_size = config.text_decoder.vocab_size
            self.text_decoder = self._build_text_decoder_model(config.text_decoder)
        if self.add_audio_encoder:
            self.audio_encoder = self._build_audio_encoder_model(config.audio_encoder)

    def shared_embedding_or_output_weight(self):
        """
        This is a convenience method to surface the language model's word embeddings, which is
        necessary for 'finalize_model_grads._allreduce_word_embedding_grads'.
        """
        if self.add_text_decoder:
            return self.text_decoder.shared_embedding_or_output_weight()
        return None

    def _build_image_encoder_model(self, config):
        vit_layer_spec = get_vit_layer_spec(config.vision_encoder)
        proj_layer_spec = get_projector_layer_spec(config.vision_projector)

        if self.pp_size <= 1:
            return VisionModel(
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
        return VisionModel(
            config=config,
            encoder_transformer_layer_spec=vit_layer_spec,
            projector_layer_spec=proj_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
        )


    def _build_audio_encoder_model(self, config):
        audio_layer_spec = get_audio_layer_spec(config.audio_encoder)

        if self.pp_size <= 1:
            return AudioModel(
                config=config,
                encoder_transformer_layer_spec=audio_layer_spec
            )
        if self.enable_vp:
            if self.pp_size * self.vp_size != len(config.audio_encoder.pipeline_num_layers) * len(
                    config.audio_encoder.pipeline_num_layers[0]):
                raise ValueError(
                    f"The product of pipeline-model-parallel-size and vpp-size must equal to the total number of stage in audio_encoder.pipeline_num_layers, "
                    f"but got pipeline-model-parallel-size: {self.pp_size}, vpp-size: {self.vp_size}, "
                    f"and total number of stage in audio_encoder.pipeline_num_layers: {len(config.audio_encoder.pipeline_num_layers) * len(config.audio_encoder.pipeline_num_layers[0])}.")
        elif self.pp_size != len(config.audio_encoder.pipeline_num_layers):
            raise ValueError(
                f"length of audio_encoder.pipeline_num_layers must equal to pipeline-model-parallel-size, "
                f"but got audio_encoder.pipeline_num_layers length:{len(config.audio_encoder.pipeline_num_layers)} "
                f"and pipeline-model-parallel-size:{self.pp_size}.")

        if self.enable_vp:
            local_num_layers = config.audio_encoder.pipeline_num_layers[self.vp_rank][self.pp_rank]
        else:
            local_num_layers = config.audio_encoder.pipeline_num_layers[self.pp_rank]

        if local_num_layers == 0:
            self.add_audio_encoder = False
            return None

        if self.enable_vp:
            pipeline_start_index = sum(
                sum(vp_layer) for vp_layer in config.audio_encoder.pipeline_num_layers[:self.vp_rank]) + sum(
                config.audio_encoder.pipeline_num_layers[self.vp_rank][:self.pp_rank])
            pipeline_end_index = sum(
                sum(vp_layer) for vp_layer in config.audio_encoder.pipeline_num_layers[:self.vp_rank]) + sum(
                config.audio_encoder.pipeline_num_layers[self.vp_rank][:self.pp_rank + 1])
        else:
            pipeline_start_index = sum(config.audio_encoder.pipeline_num_layers[:self.pp_rank])
            pipeline_end_index = sum(config.audio_encoder.pipeline_num_layers[:self.pp_rank + 1])

        pre_process = pipeline_start_index == 0
        post_process = pipeline_end_index == config.audio_encoder.num_layers
        
        print(
            f"image encoder pipeline config:\
            pp_rank:{self.pp_rank},\
            pre_process:{pre_process},\
            post_process:{post_process},\
            local_num_layers:{local_num_layers}"
        )
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        config.audio_encoder.num_layers = self.pp_size * local_num_layers
        if self.enable_vp:
            config.audio_encoder.num_layers *= self.vp_size
        return AudioModel(
            config=config,
            encoder_transformer_layer_spec=audio_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
        )

        
    def _build_text_decoder_model(self, config):
        if self.pp_size <= 1:
            return MMGPTModel(
                config=config,
                transformer_layer_spec=get_llm_layer_spec(config),
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=config.rope_theta if getattr(config, 'rope_theta', None) else config.rotary_base,
                pre_process=self.pre_process,
                post_process=self.post_process
            )
        if self.enable_vp:
            if self.pp_size * self.vp_size != len(config.pipeline_num_layers) * len(config.pipeline_num_layers[0]):
                raise ValueError(
                    f"The product of pipeline-model-parallel-size and vpp-size must equal to the total number of stage in pipeline_num_layers, "
                    f"but got pipeline-model-parallel-size: {self.pp_size}, vpp-size: {self.vp_size}, "
                    f"and total number of stage in pipeline_num_layers: {len(config.pipeline_num_layers) * len(config.pipeline_num_layers[0])}.")
        elif self.pp_size != len(config.pipeline_num_layers):
            raise ValueError(f"length of pipeline_num_layers must equal to pipeline-model-parallel-size, "
                             f"but got pipeline_num_layers length:{len(config.pipeline_num_layers)} "
                             f"and pipeline-model-parallel-size:{self.pp_size}.")

        if self.enable_vp:
            local_num_layers = config.pipeline_num_layers[self.vp_rank][self.pp_rank]
        else:
            local_num_layers = config.pipeline_num_layers[self.pp_rank]

        if local_num_layers == 0:
            self.add_text_decoder = False
            return None

        if self.enable_vp:
            pipeline_start_index = sum(
                sum(vp_layer) for vp_layer in config.pipeline_num_layers[:self.vp_rank]) + sum(
                config.pipeline_num_layers[self.vp_rank][:self.pp_rank])
            pipeline_end_index = sum(sum(vp_layer) for vp_layer in config.pipeline_num_layers[:self.vp_rank]) + sum(
                config.pipeline_num_layers[self.vp_rank][:self.pp_rank + 1])
        else:
            pipeline_start_index = sum(config.pipeline_num_layers[:self.pp_rank])
            pipeline_end_index = sum(config.pipeline_num_layers[:self.pp_rank + 1])
        
        pre_process = pipeline_start_index == 0
        post_process = pipeline_end_index == config.num_layers

        print(
            f"text decoder pipeline config:\
            pp_rank:{self.pp_rank},\
            pre_process:{pre_process},\
            post_process:{post_process},\
            local_num_layers:{local_num_layers}"
        )
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        config.num_layers = self.pp_size * local_num_layers
        if self.enable_vp:
            config.num_layers *= self.vp_size
        return MMGPTModel(
                config=config,
                transformer_layer_spec=get_llm_layer_spec(config),
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=config.rope_theta if getattr(config, 'rope_theta', None) else config.rotary_base,
                pre_process=pre_process,
                post_process=post_process
            )


    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if not len(input_tensor) == 1:
            raise AssertionError("input_tensor should only be length 1 for vlmodel")
        if self.add_image_encoder:
            self.image_encoder.set_input_tensor(input_tensor[0])
        elif self.add_text_decoder:
            if self.text_decoder.pre_process:
                self.input_tensor = input_tensor[0]
            else:
                self.text_decoder.set_input_tensor(input_tensor[0])


    def freeze(
            self,
            freeze_text_decoder: bool = False,
            freeze_image_encoder: bool = False,
            freeze_audio_encoder: bool = False,
            freeze_audio_projection: bool = False,
            freeze_image_projection: bool = False,
    ):
        """
        Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_text_decoder (bool): Freeze the text decoder module.
            freeze_image_encoder (bool): Freeze the image encoder module.
            freeze_image_projection (bool): Freeze the image projector module.
        """
        if self.add_image_encoder:
            self.image_encoder.freeze(freeze_image_encoder, freeze_image_projection)
        if self.add_audio_encoder:
            self.audio_encoder.freeze(freeze_audio_encoder, freeze_audio_projection)
        if self.add_text_decoder and freeze_text_decoder:
            for param in self.text_decoder.parameters():
                param.requires_grad = False


    def compute_loss(self, logits, labels, ignore_flag=False):
        # shift tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        if ignore_flag:
            loss = loss * 0.0

        return loss


    def compute_megatron_loss(self, logits, labels):
        # shift tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        #如果想和torch.nn.CrossEntropyLoss对齐，需要将vocab_parallel_cross_entropy中的最大值归一化代码注释掉
        loss = tensor_parallel.vocab_parallel_cross_entropy(shift_logits.float(), shift_labels) 
        loss = loss * (shift_labels > -1)
        loss = torch.sum(loss) / torch.sum(shift_labels > -1)

        return loss


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
            *args, **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:

        # MM_GRPO use, if llm_only is True, directly get vit_embeds
        if self.add_image_encoder and self.image_encoder.pre_process and kwargs.get('llm_only', False):
            vit_embeds = kwargs.get('vit_embeds').unsqueeze(1)
        elif self.add_image_encoder and pixel_values is not None:
            vit_embeds = self.image_encoder(pixel_values, image_grid_thw)
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
            input_embeds = None
            if self.text_decoder.pre_process:
                input_embeds = self.text_decoder.embedding(input_ids=input_ids, position_ids=position_ids).clone()

                if vit_embeds is not None:
                    if self.config.sequence_parallel:
                        input_embeds = gather_from_sequence_parallel_region(input_embeds)
                    input_embeds = input_embeds.transpose(0, 1)  # bsh
                    image_mask = torch.eq(input_ids, self.img_context_token_id).unsqueeze(-1).expand_as(input_embeds)
                    vit_embeds = vit_embeds[:, 0, :]
                    input_embeds = input_embeds.masked_scatter(image_mask, vit_embeds)
                    # 音频模态处理
                    if 'input_features' in kwargs:  # 使用WhisperFeatureExtractor提取音频特征后输出值名为input_feature
                        audio_features = self.audio_encoder(kwargs['input_features'], kwargs['feature_attention_mask'])
                        # 151646 表示音频模态的token id
                        audio_mask = torch.eq(input_ids, 151646).unsqueeze(-1).expand_as(input_embeds) 
                        audio_features = audio_features.to(input_embeds.device, input_embeds.dtype)
                        input_embeds = input_embeds.masked_scatter(audio_mask, audio_features)
                    input_embeds = input_embeds.transpose(0, 1)
                    if self.config.sequence_parallel:
                        input_embeds = scatter_to_sequence_parallel_region(input_embeds)

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
            )

            if self.text_decoder.post_process:
                output = output.contiguous().float()
                loss = None
                if labels is not None:
                    # if use TP then must use compute_megatron_loss, if do not use TP, then two loss are ok, but they are not equal
                    global_args = get_args()
                    if global_args.tensor_model_parallel_size > 1:
                        loss = self.compute_megatron_loss(output, labels)
                    else:
                        loss = self.compute_loss(output, labels)
                return {
                    "loss": loss,
                    "logits": output
                }
        return output
