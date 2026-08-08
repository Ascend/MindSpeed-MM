# Modified from Latte (datasets/video_transforms.py)

import random

import torch
import numpy as np


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("Clip should be Tensor, but it is %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("Clip should be 4D, but it is %dD" % clip.dim())

    return True


def to_tensor(clip):
    """Convert clip tensor (T, C, H, W) from uint8 to float (divide by 255.0) and permute dims."""
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError(
            "Clip tensor should have data type uint8, but it is %s" % str(clip.dtype)
        )

    return clip.float() / 255.0


def crop(clip, i, j, h, w):
    """Crop video clip tensor (T, C, H, W)."""
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]


def resize(clip, target_size, interpolation_mode, align_corners=False, antialias=False):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    return torch.nn.functional.interpolate(
        clip,
        size=target_size,
        mode=interpolation_mode,
        align_corners=align_corners,
        antialias=antialias,
    )


def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)


def center_crop_th_tw(clip, th, tw, top_crop):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")

    h, w = clip.size(-2), clip.size(-1)
    tr = th / tw
    if h / w > tr:
        new_h = int(w * tr)
        new_w = w
    else:
        new_h = h
        new_w = int(h / tr)

    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    return crop(clip, i, j, new_h, new_w)


def longsideresize(h, w, size, skip_low_resolution):
    if h <= size[0] and w <= size[1] and skip_low_resolution:
        return h, w

    if h / w > size[0] / size[1]:
        # hxw 720x1280  size 320x640  hw_raito 9/16 > size_ratio 8/16  neww=320/720*1280=568  newh=320
        w = int(size[0] / h * w)
        h = size[0]
    else:
        # hxw 720x1280  size 480x640  hw_raito 9/16 < size_ratio 12/16   newh=640/1280*720=360 neww=640
        # hxw 1080x1920  size 720x1280  hw_raito 9/16 = size_ratio 9/16   newh=1280/1920*1080=720 neww=1280
        h = int(size[1] / w * h)
        w = size[1]
    return h, w


def shortsideresize(h, w, size, skip_low_resolution):
    if h <= size[0] and w <= size[1] and skip_low_resolution:
        return h, w

    if h / w < size[0] / size[1]:
        w = int(size[0] / h * w)
        h = size[0]
    else:
        h = int(size[1] / w * h)
        w = size[1]
    return h, w


def calculate_centered_alignment(h: int, w: int, stride: int) -> tuple:
    """Compute centered crop offsets and stride-aligned dimensions; returns (v_off, h_off, aligned_h, aligned_w)."""
    # Calculate aligned dimensions
    aligned_h = h // stride * stride
    aligned_w = w // stride * stride

    # Compute centering offsets
    vertical_offset = (h - aligned_h) // 2
    horizontal_offset = (w - aligned_w) // 2

    return (vertical_offset, horizontal_offset, aligned_h, aligned_w)


def maxhwresize(ori_height, ori_width, max_hxw):
    if ori_height * ori_width > max_hxw:
        scale_factor = np.sqrt(max_hxw / (ori_height * ori_width))
        new_height = int(ori_height * scale_factor)
        new_width = int(ori_width * scale_factor)
    else:
        new_height = ori_height
        new_width = ori_width
    return new_height, new_width


class ToTensorVideo:
    """Convert clip from uint8 to float (divide by 255.0) and permute dims."""

    def __init__(self):
        pass

    @staticmethod
    def __call__(clip):
        """Convert clip (T, C, H, W) uint8 -> float."""
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class CenterCropResizeVideo:
    """Center-crop the short side, then resize to the specified size."""

    def __init__(
            self,
            transform_size=None,
            use_short_edge=False,
            top_crop=False,
            interpolation_mode="bilinear",
            align_corners=False,
            antialias=False,
    ):
        if transform_size is None or "max_height" not in transform_size or "max_width" not in transform_size:
            raise ValueError("Missing required param: max_height or max_width in data transform.")
        self.size = (transform_size["max_height"], transform_size["max_width"])
        self.use_short_edge = use_short_edge
        self.top_crop = top_crop
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, clip):
        """Return scale-resized / center-cropped clip (T, C, crop_size, crop_size)."""
        if self.use_short_edge:
            clip_center_crop = center_crop_using_short_edge(clip)
        else:
            clip_center_crop = center_crop_th_tw(
                clip, self.size[0], self.size[1], top_crop=self.top_crop
            )

        clip_center_crop_resize = resize(
            clip_center_crop,
            target_size=self.size,
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class ResizeVideo:
    def __init__(
        self,
        transform_size="auto",
        interpolation_mode="bilinear",
        skip_low_resolution=False,
        align_corners=False,
        antialias=False,
        mode="resize"  # resize / longside / shortside / hxw
    ):
        self.mode = mode
        if mode == 'hxw':
            self.transform_size = transform_size["max_hxw"] if isinstance(transform_size, dict) else transform_size
        elif mode in ["resize", "longside", "shortside"]:
            self.transform_size = (transform_size["max_height"], transform_size["max_width"]) if isinstance(transform_size, dict) else transform_size
        else:
            raise NotImplementedError(f"ResizeVideo only support mode `resize` / `longside` / `shortside` / `hxw`, {mode} is not implemented.")

        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.antialias = antialias
        self.skip_low_resolution = skip_low_resolution
        if isinstance(self.transform_size, str) and self.transform_size == "auto":
            raise ValueError("transform_size='auto' is a placeholder and must be replaced via get_transforms.")

    def __call__(self, clip):
        """Return scale-resized clip (T, C, H, W)."""
        h, w = clip.shape[-2:]
        if self.mode == "hxw":
            tr_h, tr_w = maxhwresize(h, w, self.transform_size)
        elif self.mode == "resize":
            tr_h, tr_w = self.transform_size
        elif self.mode == "longside":
            tr_h, tr_w = longsideresize(h, w, self.transform_size, skip_low_resolution=self.skip_low_resolution)
        elif self.mode == "shortside":
            tr_h, tr_w = shortsideresize(h, w, self.transform_size, skip_low_resolution=self.skip_low_resolution)

        if h == tr_h and w == tr_w:
            return clip
        resize_clip = resize(
            clip,
            target_size=(tr_h, tr_w),
            interpolation_mode=self.interpolation_mode,
            align_corners=self.align_corners,
            antialias=self.antialias
        )
        return resize_clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.transform_size}, interpolation_mode={self.interpolation_mode})"


class TemporalRandomCrop:
    """Temporally crop frame indices at a random location to ``size`` frames."""

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index


low_aesthetic_score_notices_video = [
    "This video has a low aesthetic quality.",
    "The beauty of this video is minimal.",
    "This video scores low in aesthetic appeal.",
    "The aesthetic quality of this video is below average.",
    "This video ranks low for beauty.",
    "The artistic quality of this video is lacking.",
    "This video has a low score for aesthetic value.",
    "The visual appeal of this video is low.",
    "This video is rated low for beauty.",
    "The aesthetic quality of this video is poor.",
]

high_aesthetic_score_notices_video = [
    "This video has a high aesthetic quality.",
    "The beauty of this video is exceptional.",
    "This video scores high in aesthetic value.",
    "With its harmonious colors and balanced composition.",
    "This video ranks highly for aesthetic quality",
    "The artistic quality of this video is excellent.",
    "This video is rated high for beauty.",
    "The aesthetic quality of this video is impressive.",
    "This video has a top aesthetic score.",
    "The visual appeal of this video is outstanding.",
]

low_aesthetic_score_notices_image = [
    "This image has a low aesthetic quality.",
    "The beauty of this image is minimal.",
    "This image scores low in aesthetic appeal.",
    "The aesthetic quality of this image is below average.",
    "This image ranks low for beauty.",
    "The artistic quality of this image is lacking.",
    "This image has a low score for aesthetic value.",
    "The visual appeal of this image is low.",
    "This image is rated low for beauty.",
    "The aesthetic quality of this image is poor.",
]

high_aesthetic_score_notices_image = [
    "This image has a high aesthetic quality.",
    "The beauty of this image is exceptional",
    "This photo scores high in aesthetic value.",
    "With its harmonious colors and balanced composition.",
    "This image ranks highly for aesthetic quality.",
    "The artistic quality of this photo is excellent.",
    "This image is rated high for beauty.",
    "The aesthetic quality of this image is impressive.",
    "This photo has a top aesthetic score.",
    "The visual appeal of this image is outstanding.",
]


def add_aesthetic_notice_video(caption, aesthetic_score):
    if aesthetic_score <= 4.25:
        notice = random.choice(low_aesthetic_score_notices_video)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    if aesthetic_score >= 5.75:
        notice = random.choice(high_aesthetic_score_notices_video)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    return caption


def add_aesthetic_notice_image(caption, aesthetic_score):
    if aesthetic_score <= 4.25:
        notice = random.choice(low_aesthetic_score_notices_image)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    if aesthetic_score >= 5.75:
        notice = random.choice(high_aesthetic_score_notices_image)
        return random.choice([caption + ' ' + notice, notice + ' ' + caption])
    return caption
