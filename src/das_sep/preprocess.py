from __future__ import annotations

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F


MAT_KEY_CANDIDATES = ["data", "Data", "signal", "x", "X"]


def find_mat_key(mat_dict: dict) -> str:
    for key in MAT_KEY_CANDIDATES:
        if key in mat_dict:
            return key
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    if not keys:
        raise KeyError("No valid matrix variable found in .mat file.")
    return keys[0]


def temporal_difference(x: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=1, prepend=x[:, :1])


def moving_average_np(x: np.ndarray, kernel_size: int = 1) -> np.ndarray:
    if kernel_size is None or kernel_size <= 1:
        return x
    kernel_size = int(kernel_size)
    pad = kernel_size // 2
    weight = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    y = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        y[c] = np.convolve(np.pad(x[c], (pad, pad), mode="edge"), weight, mode="valid")[: x.shape[1]]
    return y


def crop_or_pad(x: torch.Tensor, chunk_size: int = 10000, random_crop: bool = True) -> torch.Tensor:
    c, t = x.shape
    if t == chunk_size:
        return x
    if t > chunk_size:
        start = torch.randint(0, t - chunk_size + 1, (1,)).item() if random_crop else (t - chunk_size) // 2
        return x[:, start : start + chunk_size]
    return F.pad(x, (0, chunk_size - t))


def normalize_amp(x: torch.Tensor, amp_range: float = 0.01, per_channel: bool = False) -> torch.Tensor:
    scale = x.abs().amax(dim=1, keepdim=True) if per_channel else x.abs().max()
    return x / (scale + 1e-8) * amp_range


def normalize_zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)


def normalize_minmax(x: torch.Tensor) -> torch.Tensor:
    xmin = x.amin()
    xmax = x.amax()
    return (x - xmin) / (xmax - xmin + 1e-8)


def load_das_mat(
    path: str,
    key: str | None = None,
    use_diff: bool = True,
    norm_mode: str = "amp",
    amp_range: float = 0.01,
    channel_first: bool = True,
    per_channel_amp: bool = False,
    smooth_kernel: int = 1,
) -> torch.Tensor:
    """Load one DAS .mat sample.

    Dataset samples are normally [10000, 12] = [time, spatial channel].
    The default return is [12, T] = [spatial channel, time].
    """
    mat = sio.loadmat(path)
    key = find_mat_key(mat) if key is None else key
    arr = np.asarray(mat[key]).astype(np.float32)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D DAS matrix in {path}, got {arr.shape}")

    if arr.shape[0] >= arr.shape[1]:
        arr = arr.T
    arr = arr - arr.mean(axis=1, keepdims=True)

    if use_diff:
        arr = temporal_difference(arr)
    arr = moving_average_np(arr, smooth_kernel)

    x = torch.from_numpy(arr.copy()).float()
    if norm_mode == "amp":
        x = normalize_amp(x, amp_range=amp_range, per_channel=per_channel_amp)
    elif norm_mode == "zscore":
        x = normalize_zscore(x)
    elif norm_mode == "minmax":
        x = normalize_minmax(x)
    elif norm_mode == "none":
        pass
    else:
        raise ValueError("norm_mode must be amp, zscore, minmax, or none")

    return x if channel_first else x.transpose(0, 1)


def to_cnn_input(x: torch.Tensor) -> torch.Tensor:
    """Convert [12, T] or [B, 12, T] to [B, 1, T, 12]."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = x.transpose(1, 2)
    xmin = x.amin(dim=(1, 2), keepdim=True)
    xmax = x.amax(dim=(1, 2), keepdim=True)
    x = (x - xmin) / (xmax - xmin + 1e-8)
    return x.unsqueeze(1)


def moving_average_torch(x: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    if kernel_size is None or kernel_size <= 1:
        return x
    original_shape = x.shape
    if x.dim() == 2:
        y = x.unsqueeze(0)
    elif x.dim() == 3:
        y = x
    elif x.dim() == 4:
        b, k, c, t = x.shape
        y = x.reshape(b * k, c, t)
    else:
        raise RuntimeError(f"Unsupported shape for moving_average_torch: {x.shape}")
    c = y.shape[1]
    pad = kernel_size // 2
    weight = torch.ones(c, 1, kernel_size, dtype=y.dtype, device=y.device) / float(kernel_size)
    y = F.pad(y, (pad, pad), mode="reflect")
    y = F.conv1d(y, weight, groups=c)[..., : original_shape[-1]]
    if len(original_shape) == 2:
        return y.squeeze(0)
    if len(original_shape) == 3:
        return y
    return y.reshape(original_shape)
