import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio


MAT_KEY_CANDIDATES = ["data", "Data", "signal", "x", "X"]


def find_mat_key(mat_dict):
    for key in MAT_KEY_CANDIDATES:
        if key in mat_dict:
            return key
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    if len(keys) == 0:
        raise KeyError("No valid variable found in .mat file.")
    return keys[0]


def crop_or_pad(x, chunk_size=10000, random_crop=True):
    """
    x: torch.Tensor [C, T]
    return: [C, chunk_size]
    """
    c, t = x.shape
    if t == chunk_size:
        return x
    if t > chunk_size:
        if random_crop:
            start = torch.randint(0, t - chunk_size + 1, (1,)).item()
        else:
            start = (t - chunk_size) // 2
        return x[:, start:start + chunk_size]
    return F.pad(x, (0, chunk_size - t))


def temporal_difference(x):
    """
    x: numpy array [C, T]
    DAS论文中常用差分增强扰动变化特征。
    """
    return np.diff(x, axis=1, prepend=x[:, :1])


def moving_average_np(x, kernel_size=1):
    """Simple per-channel moving-average smoothing for high-frequency artifacts."""
    if kernel_size is None or kernel_size <= 1:
        return x
    kernel_size = int(kernel_size)
    pad = kernel_size // 2
    weight = np.ones(kernel_size, dtype=np.float32) / kernel_size
    y = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        y[c] = np.convolve(np.pad(x[c], (pad, pad), mode="edge"), weight, mode="valid")[: x.shape[1]]
    return y


def moving_average_torch(x, kernel_size=1):
    """
    x: [C, T], [B, C, T], or [B, K, C, T]
    Moving-average smoothing along time dimension.
    """
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
    y = F.conv1d(y, weight, groups=c)
    y = y[..., : original_shape[-1]]

    if len(original_shape) == 2:
        return y.squeeze(0)
    if len(original_shape) == 3:
        return y
    return y.reshape(original_shape)


def normalize_amp(x, amp_range=0.01, per_channel=False):
    if per_channel:
        scale = x.abs().amax(dim=1, keepdim=True) + 1e-8
    else:
        scale = x.abs().max() + 1e-8
    return x / scale * amp_range


def normalize_zscore(x):
    x = x - x.mean(dim=1, keepdim=True)
    x = x / (x.std(dim=1, keepdim=True) + 1e-8)
    return x


def normalize_minmax(x):
    xmin = x.amin()
    xmax = x.amax()
    return (x - xmin) / (xmax - xmin + 1e-8)


def load_das_mat(
    path,
    key=None,
    use_diff=True,
    norm_mode="amp",
    amp_range=0.01,
    channel_first=True,
    per_channel_amp=False,
    smooth_kernel=1,
):
    """
    读取BJTUSensor DAS .mat样本。

    原始常见格式: [10000, 12] = [time, spatial_channel]
    返回默认格式: [12, 10000] = [spatial_channel, time]

    norm_mode:
        "amp"    : 缩放到 [-amp_range, amp_range]，推荐用于Conv-TasNet分离
        "zscore" : 每通道零均值单位方差
        "minmax" : [0, 1]，推荐用于CNN分类
        "none"   : 只转float和去均值
    """
    mat = sio.loadmat(path)
    if key is None:
        key = find_mat_key(mat)

    arr = mat[key].astype(np.float32)

    # 兼容 [T, C] 或 [C, T]
    if arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D DAS matrix in {path}, got shape {arr.shape}")

    # 数据集样本通常是 [10000, 12]，转为 [12, 10000]
    if arr.shape[0] >= arr.shape[1]:
        arr = arr.T

    # 去直流偏置
    arr = arr - arr.mean(axis=1, keepdims=True)

    if use_diff:
        arr = temporal_difference(arr)

    if smooth_kernel and smooth_kernel > 1:
        arr = moving_average_np(arr, kernel_size=smooth_kernel)

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

    if not channel_first:
        x = x.transpose(0, 1)  # [T, C]
    return x


def to_cnn_input(x):
    """
    将DAS矩阵转为CNN输入。
    x: [12, T] or [B, 12, T]
    return: [B, 1, T, 12]
    """
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = x.transpose(1, 2)  # [B, T, 12]
    xmin = x.amin(dim=(1, 2), keepdim=True)
    xmax = x.amax(dim=(1, 2), keepdim=True)
    x = (x - xmin) / (xmax - xmin + 1e-8)
    return x.unsqueeze(1)
