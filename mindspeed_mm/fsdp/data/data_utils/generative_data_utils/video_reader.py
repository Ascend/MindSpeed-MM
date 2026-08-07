from typing import Optional, Union, Literal
from abc import ABC, abstractmethod
from pathlib import Path
import typing

import torch
import torchvision
import numpy as np
from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.utils import DecordInit
from mindspeed_mm.fsdp.utils.register import Register


VideoLayoutType = Literal["THWC", "TCHW"]
VideoArrayType = Literal["numpy", "torch"]

video_reader_register = Register()

_VIDEO_READER_ALIASES = {
    "torchvision": "TorchvisionVideo",
    "torchvisionvideo": "TorchvisionVideo",
    "decord": "DecordVideo",
    "decordvideo": "DecordVideo",
}


class Video(ABC):
    """Abstract base class defining the common video processing interface."""
    def __init__(self, video_path: str, layout: VideoLayoutType = "TCHW", array_type: VideoArrayType = "torch"):
        """Open video source; ``layout`` selects tensor layout (TCHW/THWC), ``array_type`` selects torch/numpy."""
        self.video_path = Path(video_path)
        self.layout = layout
        self.array_type = array_type
        self._validate_params()
        self._load_data()

    def _validate_params(self):
        """param validation"""
        if self.layout not in typing.get_args(VideoLayoutType):
            raise ValueError(f"Invalid video layout type: {self.layout}")
        if self.array_type not in typing.get_args(VideoArrayType):
            raise ValueError(f"Invalid video array type: {self.array_type}")

    @abstractmethod
    def _load_data(self):
        """Implementation-specific data loading; raises VideoLoadError on failure."""

    @abstractmethod
    def get_batch(self, frame_indices: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Retrieve video data batch in implementation-specific format."""

    @abstractmethod
    def get_video_fps(self) -> float:
        """Return frames per second as float."""

    @abstractmethod
    def get_len(self) -> int:
        """Get the number of frames."""


@video_reader_register.register("DecordVideo")
class DecordVideo(Video):
    """Decord-based video decoder with a shared class-level decoder instance."""
    _decoder: Optional[object] = None

    @classmethod
    def _init_decoder(cls):
        """Initialize the shared decoder once; subsequent calls reuse it."""
        if cls._decoder is None:
            cls._decoder = DecordInit()

    def _load_data(self):
        self._init_decoder()  # making sure that decoder has been initialized
        self.vframes = self.__class__._decoder(str(self.video_path))

    def get_batch(self, frame_indices: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        video_data = self.vframes.get_batch(frame_indices).asnumpy()
        if self.layout == "TCHW":
            # THWC -> TCHW,  [T: temporal, C: channel, H: height, W: width]
            video_data = video_data.transpose(0, 3, 1, 2)

        if self.array_type == "torch":
            video_data = torch.from_numpy(video_data)

        return video_data

    def get_video_fps(self) -> float:
        return self.vframes.get_avg_fps()

    def get_len(self) -> int:
        return len(self.vframes)

    def get_frame_timestamp(self, frame_index: int) -> float:
        return self.vframes.get_frame_timestamp(frame_index)

    def next(self):
        return self.vframes.next()


@video_reader_register.register("TorchvisionVideo")
class TorchvisionVideo(Video):
    """Torchvision-based video reader implementation"""
    def _load_data(self):
        self.vframes, _, self.metadata = torchvision.io.read_video(
            str(self.video_path), pts_unit="sec", output_format=self.layout
        )

    def get_batch(self, frame_indices: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        video_data = self.vframes[frame_indices]

        if self.array_type == "numpy":
            video_data = video_data.numpy()
        return video_data

    def get_video_fps(self) -> float:
        return self.metadata.get("video_fps")

    def get_len(self) -> int:
        return len(self.vframes)


class VideoReader:
    """
    Factory class for creating video reader instances.
    """
    def __init__(self, video_reader_type=None):
        """Initialize with the specified registered video backend type (e.g. decord)."""
        if video_reader_type is None:
            video_reader_type = "TorchvisionVideo"
        if isinstance(video_reader_type, str):
            video_reader_type = _VIDEO_READER_ALIASES.get(video_reader_type.lower(), video_reader_type)
        self._reader_cls = video_reader_register.get(video_reader_type)

    def __call__(self, video_path, layout: VideoLayoutType = "TCHW", array_type: VideoArrayType = "torch"):
        """Create a video reader for ``video_path`` with the given layout (TCHW default) and array_type (torch default)."""
        return self._reader_cls(video_path, layout=layout, array_type=array_type)
