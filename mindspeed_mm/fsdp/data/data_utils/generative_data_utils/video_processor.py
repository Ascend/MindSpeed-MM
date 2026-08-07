import os
import random
from collections import Counter
from typing import Optional, List
from abc import ABC, abstractmethod

import numpy as np
import torch

from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.data_transform import (
    calculate_centered_alignment,
    TemporalRandomCrop,
    maxhwresize
)
from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.transform_pipeline import get_transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.utils import (
    DataStats,
    VID_EXTENSIONS,
    TENSOR_EXTENSIONS
)
from mindspeed_mm.fsdp.utils.register import Register

video_processor_register = Register()


class VideoProcessor:
    """Factory class for creating video processor instances."""
    @staticmethod
    def create(video_processor_type=None, **kwargs) -> "AbstractVideoProcessor":
        """Initialize with the specified registered video processor type."""
        processor_cls = video_processor_register.get(video_processor_type)
        return processor_cls(**kwargs)


class AbstractVideoProcessor(ABC):
    """Base class for video processing pipelines (frame sampling + augmentation pipeline)."""

    def __init__(
        self,
        num_frames: int = 16,
        frame_interval: int = 1,
        train_pipeline: callable = None,
    ):
        """Initialize common parameters for all processors"""
        # Core sampling parameters
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.train_pipeline = train_pipeline

        # Shared components
        self.video_transforms = None  # Will be initialized per video
        self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)


    @abstractmethod
    def __call__(self, vframes, **kwargs):
        """Process video frames and return processed video data."""
        ...

    @abstractmethod
    def select_valid_data(self, data_samples):
        """Filter valid data samples; default returns the input unchanged."""
        return data_samples


@video_processor_register.register("OpensoraplanVideoProcessor")
class OpensoraplanVideoProcessor(AbstractVideoProcessor):
    """Specialized processor for Opensoraplan: frame sampling with fps/interval, resolution constraints, and VAE scale alignment."""

    def __init__(
        self,
        min_num_frames: int = 29,
        train_fps: float = 24,
        auto_interval: bool = True,
        speed_factor: float = 1.0,
        drop_short_ratio: float = 1.0,
        force_resolution: bool = True,
        max_height: int = 480,
        max_width: int = 640,
        max_hxw: int = None,
        min_hxw: int = None,
        hw_stride: int = 32,
        hw_aspect_thr: float = 1.5,
        truncate_t_by_sp: bool = True,
        vae_scale_factor: Optional[List[int]] = None,
        train_sp_batch_size: int = 1,
        global_batch_size: int = 1,
        seed: int = 42,
        **base_args
    ):
        """Initialize OpenSoraPlan specific parameters"""
        super().__init__(**base_args)
        if vae_scale_factor is None:
            vae_scale_factor = [4, 8, 8]
        self.train_fps = train_fps
        self.auto_interval = auto_interval
        self.speed_factor = speed_factor
        self.drop_short_ratio = drop_short_ratio

        # Spatial parameters
        self.force_resolution = force_resolution
        self.max_height = max_height
        self.max_width = max_width
        self.max_hxw = max_hxw
        self.min_hxw = min_hxw
        self.hw_stride = hw_stride
        self.hw_aspect_thr = hw_aspect_thr
        self.hw_aspect_thr = 1.5 if self.hw_aspect_thr == 0 else self.hw_aspect_thr
        if self.max_hxw is not None and self.min_hxw is None:
            self.min_hxw = self.max_hxw // 4
        self.transform_size = {
            "max_height": self.max_height,
            "max_width": self.max_width,
            "max_hxw": self.max_hxw,
            "min_hxw": self.min_hxw
        }

        # Training configuration
        self.ae_stride_t = vae_scale_factor[0]
        if truncate_t_by_sp:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
                self.sp_size = get_parallel_state().get_cp_size()
            else:
                self.sp_size = 1
        else:
            self.sp_size = 1
        self.train_sp_batch_size = train_sp_batch_size
        self.global_batch_size = global_batch_size
        self.min_num_frames = min_num_frames

        # Randomness control
        self.generator = torch.Generator().manual_seed(seed)

        self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline,
                                                    transform_size=self.transform_size)

    def __call__(
        self,
        vframes,
        sample_num_frames=13,
        start_frame_idx=0,
        num_frames=-1,
        crop=(None, None, None, None),
        **kwargs
    ):
        """Sample frames with temporal speed adjustment and spatial validation; returns a CTHW tensor. Raises IndexError/ValueError/AssertionError on invalid input."""
        # Frame count and FPS initialization
        total_frames = vframes.get_len() if num_frames == -1 else num_frames
        fps = vframes.get_video_fps() if vframes.get_video_fps() > 0 else 30.0
        s_x, e_x, s_y, e_y = crop

        # Temporal sampling interval calculation
        if self.auto_interval:
            # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
            frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
        else:
            frame_interval = self.frame_interval

        # Generate initial frame indices
        frame_indices = np.arange(start_frame_idx, start_frame_idx + total_frames, frame_interval).astype(int)
        frame_indices = frame_indices[frame_indices < start_frame_idx + total_frames]

        # speed up through temporal subsampling
        max_speed_factor = len(frame_indices) / self.num_frames
        if self.speed_factor > 1 and max_speed_factor > 1:
            speed_factor = min(self.speed_factor, max_speed_factor)
            target_frame_count = int(len(frame_indices) / speed_factor)
            speed_frame_idx = np.linspace(
                0, len(frame_indices) - 1, target_frame_count, dtype=int
            )
            frame_indices = frame_indices[speed_frame_idx]

        # Random temporal cropping for long sequences
        if len(frame_indices) > self.num_frames:
            begin_index, end_index = self.temporal_sample(len(frame_indices))
            frame_indices = frame_indices[begin_index:end_index]

        # to find a suitable end_frame_idx, to ensure we do not need pad video
        end_frame_idx = self.find_closest_y(
            len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
        )
        if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
            raise IndexError(
                f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
            )
        frame_indices = frame_indices[:end_frame_idx]

        # Frame validation
        if sample_num_frames != len(frame_indices):
            raise ValueError(
                f"sample_num_frames ({sample_num_frames}) is not equal with frame_indices ({len(frame_indices)})"
            )
        if len(frame_indices) < self.num_frames and self.drop_short_ratio >= 1:
            raise IndexError(
                f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
            )

        # Frame extraction and processing
        video = vframes.get_batch(frame_indices)  # T C H W
        if s_y is not None:
            video = video[:, :, s_y: e_y, s_x: e_x]

        # Resolution validation
        h, w = video.shape[-2:]
        if self.force_resolution:
            if h / w > 17 / 16 or h / w < 8 / 16:
                raise AssertionError(
                    f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But the video found ratio is {round(h / w, 2)} with the shape of {video.shape}"
                )
        # TCHW -> TCHW
        video = self.video_transforms(video)
        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return video

    def find_closest_y(self, x, vae_stride_t=4, model_ds_t=1):
        if x < self.min_num_frames:
            return -1
        for y in range(x, self.min_num_frames - 1, -1):
            if (y - 1) % vae_stride_t == 0 and ((y - 1) // vae_stride_t + 1) % model_ds_t == 0:
                # 4, 8: y in [29, 61, 93, 125, 157, 189, 221, 253, 285, 317, 349, 381, 413, 445, 477, 509, ...]
                # 4, 4: y in [29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 269, 285, 301, 317, 333, 349, 365, 381, 397, 413, 429, 445, 461, 477, 493, 509, ...]
                # 8, 1: y in [33, 41, 49, 57, 65, 73, 81, 89, 97, 105]
                # 8, 2: y in [41, 57, 73, 89, 105]
                # 8, 4: y in [57, 89]
                # 8, 8: y in [57]
                return y
        return -1

    def select_valid_data(self, data_samples):
        """Filter invalid entries (missing captions/resolution), validate constraints, and compute temporal sampling indices."""
        stats = DataStats()
        valid_samples = []
        sample_sizes = []

        for sample in data_samples:
            stats.increment('total_processed')

            if not self._validate_caption(sample, stats):
                continue

            if not self._process_resolution(sample, stats):
                continue

            if not self._process_temporal(sample, stats):
                continue

            self._validate_aesthetic(sample, stats)

            # sample update
            sample_size = f'{len(sample["sample_frame_index"])}x{sample["resolution"]["sample_height"]}x{sample["resolution"]["sample_width"]}'
            sample["sample_size"] = sample_size
            sample_sizes.append(sample_size)
            valid_samples.append(sample)

        valid_samples, sample_sizes = self._apply_final_filters(valid_samples, sample_sizes, stats)

        return valid_samples

    def _validate_caption(self, sample, stats):
        cap = sample.get("cap", None)
        if cap is None:
            stats.increment("no_caption")
            return False
        else:
            return True

    def _process_resolution(self, sample, stats):
        """Handle resolution validation and processing"""
        res_info = sample.get("resolution", {})
        height, width = res_info.get("height", -1), res_info.get("width", -1)
        if height <= 0 or width <= 0:
            stats.increment("no_resolution")
            return False

        # Process resolution
        if not self.force_resolution:
            # Dynamic resolution
            tr_h, tr_w = maxhwresize(height, width, self.max_hxw)
            _, _, sample_h, sample_w = calculate_centered_alignment(tr_h, tr_w, self.hw_stride)

            if sample_h <= 0 or sample_w <= 0:
                stats.increment("resolution_mismatch")
                return False
            if sample_h * sample_w < self.min_hxw:
                stats.increment("resolution_too_small")
                return False

            is_pick = self._filter_resolution(
                sample_h,
                sample_w,
                max_h_div_w_ratio=self.hw_aspect_thr,
                min_h_div_w_ratio=1 / self.hw_aspect_thr
            )
        else:
            # Static resolution
            aspect = self.max_height / self.max_width
            is_pick = self._filter_resolution(
                height,
                width,
                max_h_div_w_ratio=self.hw_aspect_thr * aspect,
                min_h_div_w_ratio=1 / self.hw_aspect_thr * aspect
            )
            sample_h, sample_w = self.max_height, self.max_width

        if not is_pick:
            stats.increment("aspect_mismatch")
            return False

        # Update resolution
        sample["resolution"].update(dict(sample_height=sample_h, sample_width=sample_w))
        return True

    def _filter_resolution(self, h, w, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16):
        if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
            return True
        return False

    def _process_temporal(self, sample, stats):
        """Handle temporal sampling and frame indices"""
        path = sample["path"]
        ext = os.path.splitext(path)[-1].lower()

        if ext.lower() in VID_EXTENSIONS:  # video
            return self._process_video_temporal(sample, stats)
        elif ext.lower() in IMG_EXTENSIONS:  # image
            sample["sample_frame_index"] = [0]
            sample["sample_num_frames"] = 1
            return True
        elif ext.lower() in TENSOR_EXTENSIONS:  # tensor
            return True
        else:
            raise NameError(
                f"Unknown file extension {path.split('.')[-1]}"
            )

    def _process_video_temporal(self, sample, stats):
        # no fps and duration
        duration = sample.get("duration", None)
        fps = sample.get("fps", None)
        num_frames = sample.get("num_frames", None)
        if fps is None or (duration is None and num_frames is None):
            return False

        sample["num_frames"] = round(fps * duration) if num_frames is None else num_frames
        num_frames = sample["num_frames"]

        if self.auto_interval:
            # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
            frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
        else:
            frame_interval = 1.0

        start_frame_idx = sample.get("cut", [0])[0]
        sample["start_frame_idx"] = start_frame_idx
        frame_indices = np.arange(
            start_frame_idx, start_frame_idx + num_frames, frame_interval
        ).astype(int)
        frame_indices = frame_indices[frame_indices < start_frame_idx + num_frames]

        # comment out it to enable dynamic frames training
        if (
                len(frame_indices) < self.num_frames
                and torch.rand(1, generator=self.generator).item() < self.drop_short_ratio
        ):
            stats.increment('too_short')
            return False

        # too long video will be temporal-crop randomly
        if len(frame_indices) > self.num_frames:
            begin_index, end_index = self.temporal_sample(len(frame_indices))
            frame_indices = frame_indices[begin_index:end_index]

        # to find a suitable end_frame_idx, to ensure we do not need pad video
        end_frame_idx = self.find_closest_y(
            len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
        )

        # too short that can not be encoded exactly by videovae
        if end_frame_idx == -1:
            stats.increment('too_short')
            return False

        frame_indices = frame_indices[:end_frame_idx]
        sample["sample_frame_index"] = frame_indices.tolist()
        sample["sample_num_frames"] = len(sample["sample_frame_index"])
        return True

    def _validate_aesthetic(self, sample, stats):
        # no aesthetic
        if sample.get("aesthetic", None) is None or sample.get("aes", None) is None:
            stats.increment("no_aesthetic")
        else:
            stats.collect("aesthetic_score", sample.get("aesthetic", None) or sample.get("aes", None))

    def _apply_final_filters(self, data_samples, sample_sizes, stats):
        """Apply final filters"""
        counter = Counter(sample_sizes)
        filter_major_num = 4 * self.global_batch_size
        filtered = [[i, j] for i, j in zip(data_samples, sample_sizes) if counter[j] >= filter_major_num]
        stats.print_report()
        if not filtered:
            print(f"{'After filter':<25}: 0")
            return [], []
        data_samples, sample_sizes = zip(*filtered)
        print(f"{'After filter':<25}: {len(data_samples)}")

        return data_samples, sample_sizes


@video_processor_register.register("VACEVideoProcessor")
class VACEVideoProcessor(AbstractVideoProcessor):
    def __init__(self, num_frames, auto_interval, max_height, max_width, max_hxw, train_fps, speed_factor, force_resolution,
                  vae_stride=None, vae_patch_size=None, zero_start=True, keep_last=True, **kwargs):
        super().__init__(**kwargs)
        if (num_frames - 1) % 4 != 0:
            raise AssertionError("The length of the frame must be the 4x+1")
        if vae_patch_size is None:
            vae_patch_size = [1, 2, 2]
        if vae_stride is None:
            vae_stride = [4, 8, 8]
        self.downsample = tuple([x * y for x, y in zip(vae_stride, vae_patch_size)])
        self.auto_interval = auto_interval
        self.max_height = max_height
        self.max_width = max_width
        self.speed_factor = speed_factor
        self.force_resolution = force_resolution
        self.max_hxw = max_hxw
        self.min_hxw = max_hxw
        self.train_fps = train_fps
        self.zero_start = zero_start
        self.keep_last = keep_last
        if self.max_hxw == 480 * 832:
            self.seq_len = (480 * 832 / (self.downsample[1] * self.downsample[2])) * (1 + (num_frames - 1) / 4)
        elif self.max_hxw == 720 * 1280:
            self.seq_len = (720 * 1280 / (self.downsample[1] * self.downsample[2])) * (1 + (num_frames - 1) / 4)
        else:
            raise NotImplementedError(f'image_size {self.max_hxw} is not supported')
        if self.seq_len < self.min_hxw / (self.downsample[1] * self.downsample[2]):
            raise AssertionError("seq_len is too short")
        self.rng = np.random.default_rng()


    def __call__(
            self,
            *vframes,
            crop_box=None,
            **kwargs
    ):
        fps = vframes[0].get_video_fps()
        length = min([r.get_len() for r in vframes])
        frame_timestamps = [vframes[0].get_frame_timestamp(i) for i in range(length)]
        frame_timestamps = np.array(frame_timestamps, dtype=np.float32)
        h, w = list(vframes[0].get_batch((0,)).shape[2:])
        # If a crop_box exists, x1, x2, y1, y2 are set to the crop_box values; otherwise, they are set to (0, w, 0, h).
        frame_ids, (x1, x2, y1, y2), (target_height, target_weight), fps = self._get_frameid_bbox(fps, frame_timestamps, h, w, crop_box)

        # preprocess video
        videos = [reader.get_batch(frame_ids)[:, y1:y2, x1:x2, :] for reader in vframes]

        self.image_size = (target_height, target_weight)
        video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline,
                            image_size=self.image_size)
        videos = [video_transforms(video) for video in videos]
        return *videos, frame_ids, (target_height, target_weight), fps

    def _get_frameid_bbox(self, fps, frame_timestamps, h, w, crop_box):
        if self.keep_last:
            return self._get_frameid_bbox_adjust_last(fps, frame_timestamps, h, w, crop_box)
        else:
            return self._get_frameid_bbox_default(fps, frame_timestamps, h, w, crop_box)

    # return the x previous frames
    def _get_frameid_bbox_fixed(self, fps, frame_timestamps, h, w, crop_box):
        target_fps = min(fps, self.train_fps)
        duration = frame_timestamps[-1].mean()
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        ratio = h / w
        downsample_frame, downsample_height, downsample_weight = self.downsample

        area_z = min(self.seq_len, self.max_hxw / (downsample_height * downsample_weight), (h // downsample_height) * (w // downsample_weight))
        target_frame = min(
            (int(duration * target_fps) - 1) // downsample_frame + 1,
            int(self.seq_len / area_z)
        )

        # deduce target shape of the [latent video]
        target_area_z = min(area_z, int(self.seq_len / target_frame))
        target_height = round(np.sqrt(target_area_z * ratio))
        target_weight = int(target_area_z / target_height)
        target_frame = (target_frame - 1) * downsample_frame + 1
        target_height *= downsample_height
        target_weight *= downsample_weight

        # sample frame ids
        target_duration = target_frame / target_fps
        begin = 0. if self.zero_start else random.randint(0, duration - target_duration)
        timestamps = np.linspace(begin, begin + target_duration, target_frame)
        frame_ids = list(range(0, target_frame))
        return frame_ids, (x1, x2, y1, y2), (target_height, target_weight), target_fps

    # extrace a video from the target_duration and evenly capture the target_frame form it
    def _get_frameid_bbox_default(self, fps, frame_timestamps, h, w, crop_box):
        # Extract a number of frames from a specific segment of the video.
        target_fps = min(fps, self.train_fps)
        duration = frame_timestamps[-1].mean()
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        ratio = h / w
        downsample_frame, downsample_height, downsample_weight = self.downsample

        area_z = min(self.seq_len, self.max_hxw / (downsample_height * downsample_weight), (h // downsample_height) * (w // downsample_weight))
        target_frame = min(
            (int(duration * target_fps) - 1) // downsample_frame + 1,
            int(self.seq_len / area_z)
        )

        # deduce target shape of the [latent video]
        target_area_z = min(area_z, int(self.seq_len / target_frame))
        target_height = round(np.sqrt(target_area_z * ratio))
        target_weight = int(target_area_z / target_height)
        target_frame = (target_frame - 1) * downsample_frame + 1
        target_height *= downsample_height
        target_weight *= downsample_weight

        # sample frame ids
        target_duration = target_frame / target_fps
        begin = 0. if self.zero_start else random.randint(0, duration - target_duration)
        timestamps = np.linspace(begin, begin + target_duration, target_frame)
        frame_ids = np.argmax(np.logical_and(
            timestamps[:, None] >= frame_timestamps[None, :, 0],
            timestamps[:, None] < frame_timestamps[None, :, 1]
        ), axis=1).tolist()
        return frame_ids, (x1, x2, y1, y2), (target_height, target_weight), target_fps

    # evenly capture the target_frame form the video
    def _get_frameid_bbox_adjust_last(self, fps, frame_timestamps, h, w, crop_box):
        duration = frame_timestamps[-1].mean()
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        ratio = h / w
        downsample_frame, downsample_height, downsample_weight = self.downsample

        area_z = min(self.seq_len, self.max_hxw / (downsample_height * downsample_weight), (h // downsample_height) * (w // downsample_weight))
        target_frame = min(
            (len(frame_timestamps) - 1) // downsample_frame + 1,
            int(self.seq_len / area_z)
        )

        # deduce target shape of the [latent video]
        target_area_z = min(area_z, int(self.seq_len / target_frame))
        target_height = round(np.sqrt(target_area_z * ratio))
        target_weight = int(target_area_z / target_height)
        target_frame = (target_frame - 1) * downsample_frame + 1
        target_height *= downsample_height
        target_weight *= downsample_weight

        # sample frame ids
        target_duration = duration
        target_fps = target_frame / target_duration
        timestamps = np.linspace(0., target_duration, target_frame)
        frame_ids = np.argmax(np.logical_and(
            timestamps[:, None] >= frame_timestamps[None, :, 0],
            timestamps[:, None] <= frame_timestamps[None, :, 1]
        ), axis=1).tolist()
        return frame_ids, (x1, x2, y1, y2), (target_height, target_weight), target_fps

    def select_valid_data(self, data_samples):
        return super().select_valid_data(data_samples)
