import re
from types import SimpleNamespace
from typing import Union, List

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor
from transformers.audio_utils import mel_filter_bank
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from transformers.processing_utils import (
    AudioKwargs,
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    VideosKwargs,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class LongcatNextProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: ImagesKwargs
    videos_kwargs: VideosKwargs
    audio_kwargs: AudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
            "return_attention_mask": False,
        }
    }


class LongcatNextAudioProcessor(FeatureExtractionMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.num_mel_bins,
            min_frequency=0.0,
            max_frequency=self.sampling_rate / 2.0,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.window = torch.hann_window(self.n_fft)

    @staticmethod
    def zero_mean_unit_var_norm(x):
        return (x - x.mean()) / torch.sqrt(x.var() + 1e-8)

    def load_audio_waveform(self, uri, metadata=None, waveform_tensor=None, return_tensors=True, do_normalize=False):
        if metadata is None or waveform_tensor is None:
            waveform_np, sample_rate = librosa.load(uri, sr=None, mono=False)

            if waveform_np.ndim == 1:
                waveform_tensor = torch.from_numpy(waveform_np).unsqueeze(0)
            else:
                waveform_tensor = torch.from_numpy(waveform_np)

            try:
                sf_info = sf.info(uri)
                metadata = SimpleNamespace(
                    sample_rate=sample_rate,
                    num_frames=waveform_tensor.shape[1],
                    num_channels=waveform_tensor.shape[0],
                    bits_per_sample=getattr(sf_info, 'bits_per_sample', 16),
                    encoding=getattr(sf_info, 'subtype', 'PCM_F')
                )
            except Exception:
                metadata = SimpleNamespace(
                    sample_rate=sample_rate,
                    num_frames=waveform_tensor.shape[1],
                    num_channels=waveform_tensor.shape[0],
                    bits_per_sample=16,
                    encoding='PCM_F'
                )

        if not (metadata.num_channels <= 2):
            raise AssertionError("acoustic file with {} channels.".format(
                metadata.num_channels))  # whisper only accept mono channel audio

        if self.sampling_rate != metadata.sample_rate:
            waveform_tensor = torch.nn.functional.interpolate(
                waveform_tensor.unsqueeze(0),
                size=int(waveform_tensor.shape[1] * self.sampling_rate / metadata.sample_rate),
                mode='linear',
                align_corners=False
            ).squeeze(0)

        if metadata.num_channels > 1:
            waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)

        if do_normalize:
            waveform_tensor = self.zero_mean_unit_var_norm(waveform_tensor)

        if return_tensors:  # (channels, samples)
            return waveform_tensor
        else:
            return waveform_tensor.numpy()

    def split_with_overlap(self, waveform):
        channels, wave_samples = waveform.shape
        max_audio_samples = self.max_audio_seconds * self.sampling_rate
        if wave_samples <= max_audio_samples or self.split_overlap < 0:
            return [waveform]

        split_waveform, start = [], 0
        while start < wave_samples:
            if start > int(self.sampling_rate * self.split_overlap):
                start -= int(self.sampling_rate * self.split_overlap)
            end = min(start + max_audio_samples, wave_samples)
            if end - start >= self.n_fft:
                split_waveform.append(waveform[:, start:end])
            start = end
        return split_waveform

    @classmethod
    def inference_output_length(self, input_length, kernel_size, stride_size, avg_pooler):
        # for whisper + bridge
        encoder_length = (input_length + 2 * (kernel_size // 2) - kernel_size) // 1 + 1  # conv layer1 with pad=1
        encoder_length = (encoder_length + 2 * (
                kernel_size // 2) - kernel_size) // stride_size + 1  # conv layer2 with pad=1
        if avg_pooler > 1:
            bridge_length = encoder_length // avg_pooler
        return encoder_length, bridge_length

    def extract_fbank_features(self, waveform):
        channels, wave_samples = waveform.shape
        if not (wave_samples >= self.n_fft):
            raise AssertionError("error: wave_samples < self.n_fft")
        valid_frame_nums = min(self.max_audio_seconds * self.sampling_rate // self.hop_length,
                               wave_samples // self.hop_length + 1)
        if wave_samples < self.max_audio_seconds * self.sampling_rate:
            waveform = torch.nn.functional.pad(waveform,
                                               (0, self.max_audio_seconds * self.sampling_rate - wave_samples),
                                               "constant", 0)
        else:
            waveform = waveform[:, :self.max_audio_seconds * self.sampling_rate]

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=self.window,
                          return_complex=True)  # fft, len(wave) // n_fft // 2 + 1
        magnitudes = stft[..., :-1].abs() ** 2

        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        log_spec = log_spec[0].numpy()  # (channel, filters, samples) -> (filters, samples)
        log_spec[:, valid_frame_nums:] = 0.0  # pad0

        return log_spec, valid_frame_nums

    def process(self, audio_path, **kwargs):
        metadata, waveform_tensors = None, None
        waveforms = self.load_audio_waveform(audio_path, metadata, waveform_tensors, True)
        waveforms = self.split_with_overlap(waveforms)

        ret_audio, ret_encoder_length, ret_bridge_length = [], [], []
        for i, waveform in enumerate(waveforms):
            audio, input_length = self.extract_fbank_features(waveform)
            encoder_length, bridge_length = self.inference_output_length(input_length, self.kernel_size,
                                                                         self.stride_size, self.avg_pooler)
            if bridge_length <= 0:
                continue

            ret_audio.append(audio)
            ret_encoder_length.append(encoder_length)
            ret_bridge_length.append(bridge_length)
        return ret_audio, ret_encoder_length, ret_bridge_length

    def __call__(self, audio: Union[str, List[str]], **kwargs):
        if isinstance(audio, str):
            audio = [audio]
        results = {
            "audio": [],
            "encoder_length": [],
            "bridge_length": [],
        }
        for audio_path in audio:
            audio, encoder_length, bridge_length = self.process(audio_path, **kwargs)
            results["audio"].append(audio)
            results["encoder_length"].append(encoder_length)
            results["bridge_length"].append(bridge_length)
        return results


class LongcatNextProcessor(ProcessorMixin):
    attributes = ["image_processor", "video_processor", "audio_processor", "tokenizer"]

    image_processor_class = "Qwen2VLImageProcessor"
    video_processor_class = "Qwen2VLImageProcessor"
    audio_processor_class = "LongcatNextAudioProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, video_processor=None, audio_processor=None, tokenizer=None,
                 chat_template=None, **kwargs):
        super().__init__(image_processor, video_processor, audio_processor, tokenizer, chat_template=chat_template)
        init_token_list = [
            "image_start_token", "image_end_token", "image_pad_token", "image_newline_token",
            "audio_start_token", "audio_end_token", "audio_pad_token",
        ]
        for attr in init_token_list:
            token_str = self.tokenizer.init_kwargs.get(attr)
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if not len(token_ids) == 1:
                raise AssertionError(
                    f"{attr}='{token_str}' encode to get {len(token_ids)} id(s) {token_ids}, expect 1 id")
            setattr(self, f"{attr}", token_str)
            setattr(self, f"{attr}_id", token_ids[0])

    def __call__(
            self,
            text: str,
            **kwargs,
    ) -> List["LongcatNextProcessorOutput"]:

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            LongcatNextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, str):
            raise AssertionError("text type should be str")

        image_path_list = re.findall(rf"{self.image_start_token}(.*?){self.image_end_token}", text)
        audio_path_list = re.findall(rf"{self.audio_start_token}(.*?){self.audio_end_token}", text)

        if len(image_path_list) > 0:
            images_inputs = self.image_processor(images=image_path_list, **output_kwargs["images_kwargs"])
            image_grid_thw = images_inputs["image_grid_thw"]
            for i, image_path in enumerate(image_path_list):
                image_token_num = image_grid_thw[i][0] * (
                        image_grid_thw[i][1] // self.image_processor.spatial_merge_size) * (
                                          image_grid_thw[i][2] // self.image_processor.spatial_merge_size)
                text = text.replace(f"{self.image_start_token}{image_path}{self.image_end_token}",
                                    f"{self.image_start_token}{self.image_pad_token * image_token_num}{self.image_end_token}")
        else:
            images_inputs = {}

        if len(audio_path_list) > 0:
            audio_inputs = self.audio_processor(audio=audio_path_list, **output_kwargs["audio_kwargs"])
            for i, audio_path in enumerate(audio_path_list):
                audio_token_num = np.sum(audio_inputs["bridge_length"][i])
                text = text.replace(f"{self.audio_start_token}{audio_path}{self.audio_end_token}",
                                    f"{self.audio_start_token}{self.audio_pad_token * audio_token_num}{self.audio_end_token}")
            for key in audio_inputs:
                audio_inputs[key] = [val for b_val in audio_inputs[key] for val in b_val]
        else:
            audio_inputs = {}

        texts_inputs = self.tokenizer([text], **output_kwargs["text_kwargs"])

        batch_feature_func = lambda x: BatchFeature(
            data={**x},
            tensor_type=kwargs.get("return_tensors"),
        )
        return (
            batch_feature_func(texts_inputs),
            batch_feature_func({k.replace("image", "visual"): v for k, v in images_inputs.items()}) if len(
                images_inputs) > 0 else None,
            batch_feature_func(audio_inputs) if len(audio_inputs) > 0 else None,
        )


class LongcatNextAudioProcessorConfig(PretrainedConfig):
    pass


AutoFeatureExtractor.register(LongcatNextAudioProcessorConfig, LongcatNextAudioProcessor)

__all__ = ["LongcatNextAudioProcessor", "LongcatNextProcessor"]
