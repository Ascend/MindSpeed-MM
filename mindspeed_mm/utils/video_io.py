import numpy as np
import torch
from PIL import Image

import av


def channels_last(video):
    """Convert a [c, t, h, w] tensor/ndarray to [t, h, w, c]."""
    if torch.is_tensor(video):
        return video.permute(1, 2, 3, 0)
    return np.transpose(np.asarray(video), (1, 2, 3, 0))


def _to_uint8(video, value_range=(-1, 1), normalize=True):
    if torch.is_tensor(video):
        video = video.detach().cpu().numpy()
    video = np.asarray(video)
    if video.dtype == np.uint8:
        return np.ascontiguousarray(video)
    if value_range is not None and normalize:
        vmin, vmax = value_range
        video = (video.astype(np.float32) - vmin) / (vmax - vmin)
    return np.clip(video * 255.0, 0, 255).astype(np.uint8)


def save_video_mp4(video, path, fps, value_range=(-1, 1), normalize=True, crf="18"):
    """Save a [t, h, w, c] RGB video to H.264 MP4 using PyAV.

    Accepts uint8 or float input; float values are normalized from
    ``value_range`` (default [-1, 1]) to [0, 255] before encoding.
    """
    frames = _to_uint8(video, value_range, normalize)
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError("video must have shape [t, h, w, c] with c in (1, 3, 4)")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    pix_fmt = "rgb24" if frames.shape[-1] == 3 else "gray"
    container = av.open(path, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = frames.shape[2]
    stream.height = frames.shape[1]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf)}
    for frame in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format=pix_fmt)):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def save_image_png(image, path, value_range=(-1, 1), normalize=True):
    """Save a [h, w] or [h, w, c] image to PNG using PIL."""
    frames = _to_uint8(image, value_range, normalize)
    if frames.ndim == 3 and frames.shape[-1] == 1:
        frames = frames[..., 0]
    if frames.ndim == 2 or (frames.ndim == 3 and frames.shape[-1] in (3, 4)):
        Image.fromarray(frames).save(path)
    else:
        raise ValueError("image must have shape [h, w] or [h, w, c] with c in (1, 3, 4)")
