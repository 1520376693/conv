from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .preprocess import crop_or_pad, load_das_mat, to_cnn_input


CLASS_NAMES = [
    "01_background",
    "02_dig",
    "03_knock",
    "04_water",
    "05_shake",
    "06_walk",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}
EMPTY_LABEL = 6


def scan_class_files(root: str, split: str) -> Dict[int, List[str]]:
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    class_to_files = {}
    for label, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(".mat")]
        if not files:
            raise RuntimeError(f"No .mat files found in {cls_dir}")
        class_to_files[label] = sorted(files)
    return class_to_files


def random_shift_zero(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return x
    shift = random.randint(-max_shift, max_shift)
    return shift_zero(x, shift)


def shift_zero(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return x
    y = torch.zeros_like(x)
    if shift > 0:
        y[:, shift:] = x[:, :-shift]
    else:
        y[:, :shift] = x[:, -shift:]
    return y


def channel_shift_zero(x: torch.Tensor, shifts: List[int]) -> torch.Tensor:
    if not shifts or all(int(shift) == 0 for shift in shifts):
        return x
    y = torch.zeros_like(x)
    channels = min(x.shape[0], len(shifts))
    for ch in range(channels):
        shift = int(shifts[ch])
        if shift == 0:
            y[ch] = x[ch]
        elif shift > 0:
            y[ch, shift:] = x[ch, :-shift]
        else:
            y[ch, :shift] = x[ch, -shift:]
    if channels < x.shape[0]:
        y[channels:] = x[channels:]
    return y


def apply_spatial_mix_params(x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
    shifts = [int(v) for v in params.get("channel_shifts", [])]
    if shifts:
        x = channel_shift_zero(x, shifts)
    gains = params.get("per_channel_gains", [])
    if gains:
        gain_tensor = torch.as_tensor(gains[: x.shape[0]], dtype=x.dtype, device=x.device)
        if gain_tensor.numel() < x.shape[0]:
            pad = torch.ones(x.shape[0] - gain_tensor.numel(), dtype=x.dtype, device=x.device)
            gain_tensor = torch.cat([gain_tensor, pad], dim=0)
        x = x * gain_tensor[:, None]
    blend = float(params.get("spatial_blend", 0.0))
    if blend > 0:
        center = float(params.get("spatial_center", (x.shape[0] - 1) / 2.0))
        width = max(float(params.get("spatial_width", x.shape[0])), 1e-3)
        idx = torch.arange(x.shape[0], dtype=x.dtype, device=x.device)
        profile = torch.exp(-0.5 * ((idx - center) / width).pow(2))
        profile = profile / profile.max().clamp_min(1e-6)
        weights = (1.0 - blend) + blend * profile
        x = x * weights[:, None]
    return x


def add_white_noise(x: torch.Tensor, snr_db_range: Tuple[float, float]) -> torch.Tensor:
    snr_db = random.uniform(*snr_db_range)
    sig_power = x.pow(2).mean()
    noise = torch.randn_like(x)
    scale = torch.sqrt(sig_power / (10.0 ** (snr_db / 10.0)) / (noise.pow(2).mean() + 1e-8))
    return x + noise * scale


def random_time_mask(x: torch.Tensor, max_width: int = 0, prob: float = 0.0) -> torch.Tensor:
    if max_width <= 0 or random.random() > prob:
        return x
    if x.dim() == 2:
        t = x.shape[-1]
        width = random.randint(1, min(max_width, t))
        start = random.randint(0, t - width)
        y = x.clone()
        y[:, start : start + width] = 0
        return y
    if x.dim() == 3:
        t = x.shape[1]
        width = random.randint(1, min(max_width, t))
        start = random.randint(0, t - width)
        y = x.clone()
        y[:, start : start + width, :] = 0
        return y
    return x


class DASSingleEventDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        chunk_size: int = 10000,
        use_diff: bool = True,
        augment: bool = False,
        aug_gain=(0.8, 1.25),
        aug_noise_prob: float = 0.3,
        aug_snr_db=(18.0, 35.0),
        aug_shift: int = 800,
        aug_time_mask_prob: float = 0.2,
        aug_time_mask_width: int = 600,
        aug_artifact_noise_prob: float = 0.15,
        smooth_kernel: int = 1,
        max_items: int | None = None,
    ):
        self.root = root
        self.split = split
        self.chunk_size = chunk_size
        self.use_diff = use_diff
        self.augment = augment and split == "train"
        self.aug_gain = tuple(aug_gain)
        self.aug_noise_prob = aug_noise_prob
        self.aug_snr_db = tuple(aug_snr_db)
        self.aug_shift = aug_shift
        self.aug_time_mask_prob = aug_time_mask_prob
        self.aug_time_mask_width = aug_time_mask_width
        self.aug_artifact_noise_prob = aug_artifact_noise_prob
        self.smooth_kernel = smooth_kernel
        self.items = [(path, label) for label, files in scan_class_files(root, split).items() for path in files]
        if max_items is not None and max_items > 0:
            rng = random.Random(2026 if split != "train" else 2027)
            rng.shuffle(self.items)
            self.items = self.items[:max_items]

    def __len__(self):
        return len(self.items)

    @property
    def labels(self):
        return [label for _, label in self.items]

    def __getitem__(self, index):
        path, label = self.items[index]
        x = load_das_mat(path, use_diff=self.use_diff, norm_mode="minmax", smooth_kernel=self.smooth_kernel)
        x = crop_or_pad(x, self.chunk_size, random_crop=(self.split == "train"))
        if self.augment:
            x = random_shift_zero(x, self.aug_shift)
            if random.random() < self.aug_noise_prob:
                x = add_white_noise(x, self.aug_snr_db)
            if random.random() < self.aug_artifact_noise_prob:
                # Simulate weak residual artifacts produced by separation.
                x = x + 0.03 * torch.randn_like(x) * x.std().clamp_min(1e-6)
            x = x * random.uniform(*self.aug_gain)
        x = to_cnn_input(x).squeeze(0)
        if self.augment:
            x = random_time_mask(x, self.aug_time_mask_width, self.aug_time_mask_prob)
        return {"x": x.float(), "label": torch.tensor(label).long(), "path": path}


class DASRandomMixDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        chunk_size: int = 10000,
        min_sources: int = 2,
        max_sources: int = 4,
        amp_range: float = 0.01,
        allow_background: bool = False,
        deterministic: bool | None = None,
        use_diff: bool = True,
        epoch_length: int | None = None,
        two_source_prob: float = 0.45,
        three_source_prob: float = 0.35,
        same_class_prob: float = 0.10,
        gain_range=(0.55, 1.65),
        max_source_shift: int = 1200,
        polarity_flip_prob: float = 0.1,
        add_background_noise: bool = True,
        background_noise_prob: float = 0.6,
        background_snr_db=(14.0, 28.0),
        add_white_noise_prob: float = 0.15,
        white_noise_snr_db=(20.0, 35.0),
        smooth_kernel: int = 1,
        spatial_mix_prob: float = 0.0,
        channel_delay_max: int = 0,
        channel_delay_slope_range=(0.0, 0.0),
        per_channel_gain_range=(1.0, 1.0),
        spatial_center_width_range=(12.0, 12.0),
        spatial_blend_range=(0.0, 0.0),
        hard_profile_prob: float = 0.0,
        hard_two_source_prob: float | None = None,
        hard_three_source_prob: float | None = None,
        hard_same_class_prob: float | None = None,
        hard_gain_range=None,
        hard_max_source_shift: int | None = None,
        hard_polarity_flip_prob: float | None = None,
        hard_background_noise_prob: float | None = None,
        hard_spatial_mix_prob: float | None = None,
        hard_channel_delay_max: int | None = None,
        hard_channel_delay_slope_range=None,
        hard_per_channel_gain_range=None,
        hard_spatial_center_width_range=None,
        hard_spatial_blend_range=None,
    ):
        self.root = root
        self.split = split
        self.chunk_size = chunk_size
        self.min_sources = min_sources
        self.max_sources = max_sources
        self.amp_range = amp_range
        self.allow_background = allow_background
        self.deterministic = (split != "train") if deterministic is None else deterministic
        self.use_diff = use_diff
        self.epoch_length = epoch_length
        self.two_source_prob = two_source_prob
        self.three_source_prob = three_source_prob
        self.same_class_prob = same_class_prob
        self.gain_range = tuple(gain_range)
        self.max_source_shift = max_source_shift
        self.polarity_flip_prob = polarity_flip_prob
        self.add_background_noise = add_background_noise
        self.background_noise_prob = background_noise_prob
        self.background_snr_db = tuple(background_snr_db)
        self.add_white_noise_prob = add_white_noise_prob
        self.white_noise_snr_db = tuple(white_noise_snr_db)
        self.smooth_kernel = smooth_kernel
        self.spatial_mix_prob = spatial_mix_prob if split == "train" else 0.0
        self.channel_delay_max = int(channel_delay_max)
        self.channel_delay_slope_range = tuple(channel_delay_slope_range)
        self.per_channel_gain_range = tuple(per_channel_gain_range)
        self.spatial_center_width_range = tuple(spatial_center_width_range)
        self.spatial_blend_range = tuple(spatial_blend_range)
        self.hard_profile_prob = hard_profile_prob if split == "train" else 0.0
        self.hard_profile = {
            "two_source_prob": self.two_source_prob if hard_two_source_prob is None else hard_two_source_prob,
            "three_source_prob": self.three_source_prob if hard_three_source_prob is None else hard_three_source_prob,
            "same_class_prob": self.same_class_prob if hard_same_class_prob is None else hard_same_class_prob,
            "gain_range": self.gain_range if hard_gain_range is None else tuple(hard_gain_range),
            "max_source_shift": self.max_source_shift if hard_max_source_shift is None else int(hard_max_source_shift),
            "polarity_flip_prob": self.polarity_flip_prob if hard_polarity_flip_prob is None else hard_polarity_flip_prob,
            "background_noise_prob": self.background_noise_prob if hard_background_noise_prob is None else hard_background_noise_prob,
            "spatial_mix_prob": self.spatial_mix_prob if hard_spatial_mix_prob is None else hard_spatial_mix_prob,
            "channel_delay_max": self.channel_delay_max if hard_channel_delay_max is None else int(hard_channel_delay_max),
            "channel_delay_slope_range": self.channel_delay_slope_range
            if hard_channel_delay_slope_range is None
            else tuple(hard_channel_delay_slope_range),
            "per_channel_gain_range": self.per_channel_gain_range
            if hard_per_channel_gain_range is None
            else tuple(hard_per_channel_gain_range),
            "spatial_center_width_range": self.spatial_center_width_range
            if hard_spatial_center_width_range is None
            else tuple(hard_spatial_center_width_range),
            "spatial_blend_range": self.spatial_blend_range
            if hard_spatial_blend_range is None
            else tuple(hard_spatial_blend_range),
        }
        self.class_to_files = scan_class_files(root, split)
        default_length = sum(len(v) for v in self.class_to_files.values())
        self.epoch_length = epoch_length or (default_length * 2 if split == "train" else default_length)

    def __len__(self):
        return self.epoch_length

    def _choose_n_src(self, profile: Dict[str, Any] | None = None) -> int:
        two_source_prob = self.two_source_prob if profile is None else profile["two_source_prob"]
        three_source_prob = self.three_source_prob if profile is None else profile["three_source_prob"]
        if self.min_sources == self.max_sources:
            return self.max_sources
        if self.min_sources == 2 and self.max_sources >= 4:
            r = random.random()
            if r < two_source_prob:
                return 2
            if r < two_source_prob + three_source_prob:
                return 3
            return 4
        if self.min_sources == 2 and self.max_sources == 3:
            return 2 if random.random() < two_source_prob else 3
        return random.randint(self.min_sources, self.max_sources)

    def _load_signal(self, label: int) -> torch.Tensor:
        path = random.choice(self.class_to_files[label])
        sig = load_das_mat(
            path,
            use_diff=self.use_diff,
            norm_mode="amp",
            amp_range=self.amp_range,
            smooth_kernel=self.smooth_kernel,
        )
        return crop_or_pad(sig, self.chunk_size, random_crop=(self.split == "train"))

    def _choose_labels(self, n_src: int, profile: Dict[str, Any] | None = None) -> List[int]:
        same_class_prob = self.same_class_prob if profile is None else profile["same_class_prob"]
        labels = list(range(len(CLASS_NAMES)))
        if not self.allow_background:
            labels = labels[1:]
        if random.random() < same_class_prob:
            return random.choices(labels, k=n_src)
        if n_src <= len(labels):
            return random.sample(labels, n_src)
        return random.choices(labels, k=n_src)

    def _random_spatial_params(self, channels: int, profile: Dict[str, Any] | None = None) -> Dict[str, Any]:
        channel_delay_max = self.channel_delay_max if profile is None else profile["channel_delay_max"]
        channel_delay_slope_range = self.channel_delay_slope_range if profile is None else profile["channel_delay_slope_range"]
        per_channel_gain_range = self.per_channel_gain_range if profile is None else profile["per_channel_gain_range"]
        spatial_center_width_range = self.spatial_center_width_range if profile is None else profile["spatial_center_width_range"]
        spatial_blend_range = self.spatial_blend_range if profile is None else profile["spatial_blend_range"]
        max_delay = max(0, channel_delay_max)
        if max_delay > 0:
            mid = (channels - 1) / 2.0
            slope = random.uniform(*channel_delay_slope_range)
            offset = random.uniform(-max_delay, max_delay)
            channel_shifts = [
                max(-max_delay, min(max_delay, int(round(offset + (ch - mid) * slope))))
                for ch in range(channels)
            ]
        else:
            channel_shifts = [0 for _ in range(channels)]
        gain_low, gain_high = per_channel_gain_range
        if gain_low == gain_high == 1.0:
            per_channel_gains = [1.0 for _ in range(channels)]
        else:
            per_channel_gains = [random.uniform(gain_low, gain_high) for _ in range(channels)]
        width = random.uniform(*spatial_center_width_range)
        return {
            "channel_shifts": channel_shifts,
            "per_channel_gains": per_channel_gains,
            "spatial_center": random.uniform(0.0, float(channels - 1)),
            "spatial_width": width,
            "spatial_blend": random.uniform(*spatial_blend_range),
        }

    def _maybe_apply_spatial_mix(self, sig: torch.Tensor, profile: Dict[str, Any] | None = None) -> torch.Tensor:
        spatial_mix_prob = self.spatial_mix_prob if profile is None else profile["spatial_mix_prob"]
        if spatial_mix_prob <= 0 or random.random() >= spatial_mix_prob:
            return sig
        return apply_spatial_mix_params(sig, self._random_spatial_params(sig.shape[0], profile))

    def __getitem__(self, index):
        if self.deterministic:
            random.seed(index)
            torch.manual_seed(index)
        profile = self.hard_profile if random.random() < self.hard_profile_prob else None
        n_src = self._choose_n_src(profile)
        labels = self._choose_labels(n_src, profile)
        gain_range = self.gain_range if profile is None else profile["gain_range"]
        max_source_shift = self.max_source_shift if profile is None else profile["max_source_shift"]
        polarity_flip_prob = self.polarity_flip_prob if profile is None else profile["polarity_flip_prob"]
        refs, out_labels = [], []
        for label in labels:
            sig = self._load_signal(label)
            if self.split == "train":
                sig = random_shift_zero(sig, max_source_shift)
                sig = sig * random.uniform(*gain_range)
                if random.random() < polarity_flip_prob:
                    sig = -sig
                sig = self._maybe_apply_spatial_mix(sig, profile)
            refs.append(sig)
            out_labels.append(label)
        while len(refs) < self.max_sources:
            refs.append(torch.zeros_like(refs[0]))
            out_labels.append(EMPTY_LABEL)
        refs = torch.stack(refs, dim=0)
        mix = refs[:n_src].sum(dim=0)
        background_noise_prob = self.background_noise_prob if profile is None else profile["background_noise_prob"]
        if self.add_background_noise and random.random() < background_noise_prob:
            bg = self._load_signal(0)
            snr_db = random.uniform(*self.background_snr_db)
            scale = torch.sqrt(mix.pow(2).mean() / (10.0 ** (snr_db / 10.0)) / (bg.pow(2).mean() + 1e-8))
            mix = mix + bg * scale
        if self.split == "train" and random.random() < self.add_white_noise_prob:
            mix = add_white_noise(mix, self.white_noise_snr_db)
        scale = self.amp_range / (mix.abs().max() + 1e-8)
        return {
            "mix": (mix * scale).float(),
            "ref": (refs * scale).float(),
            "labels": torch.tensor(out_labels).long(),
            "n_src": torch.tensor(n_src).long(),
        }


class DASManifestMixDataset(Dataset):
    def __init__(
        self,
        root: str,
        manifest_csv: str,
        chunk_size: int = 10000,
        max_sources: int = 4,
        amp_range: float = 0.01,
        use_diff: bool = True,
        smooth_kernel: int = 1,
    ):
        self.root = root
        self.manifest_csv = manifest_csv
        self.chunk_size = chunk_size
        self.max_sources = max_sources
        self.amp_range = amp_range
        self.use_diff = use_diff
        self.smooth_kernel = smooth_kernel
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise RuntimeError(f"Manifest is empty: {manifest_csv}")

    def __len__(self):
        return len(self.rows)

    def _json(self, row: dict, key: str, default):
        raw = row.get(key, "")
        if raw == "":
            return default
        return json.loads(raw)

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.root, path)

    def _load_path(self, path: str) -> torch.Tensor:
        sig = load_das_mat(
            self._resolve_path(path),
            use_diff=self.use_diff,
            norm_mode="amp",
            amp_range=self.amp_range,
            smooth_kernel=self.smooth_kernel,
        )
        return crop_or_pad(sig, self.chunk_size, random_crop=False)

    def __getitem__(self, index):
        row = self.rows[index]
        source_paths = self._json(row, "source_paths", [])
        labels = [int(v) for v in self._json(row, "labels", [])]
        gains = [float(v) for v in self._json(row, "gains", [1.0 for _ in source_paths])]
        shifts = [int(v) for v in self._json(row, "shifts", [0 for _ in source_paths])]
        spatial_params = self._json(row, "spatial_params", [{} for _ in source_paths])
        n_src = int(row.get("n_src", len(source_paths)))
        if n_src <= 0 or n_src > self.max_sources:
            raise ValueError(f"Invalid n_src={n_src} in manifest row {index}")
        refs, out_labels = [], []
        for src_idx, path in enumerate(source_paths[:n_src]):
            sig = self._load_path(path)
            sig = shift_zero(sig, shifts[src_idx] if src_idx < len(shifts) else 0)
            sig = sig * (gains[src_idx] if src_idx < len(gains) else 1.0)
            if src_idx < len(spatial_params):
                sig = apply_spatial_mix_params(sig, spatial_params[src_idx])
            refs.append(sig)
            out_labels.append(labels[src_idx] if src_idx < len(labels) else EMPTY_LABEL)
        while len(refs) < self.max_sources:
            refs.append(torch.zeros_like(refs[0]))
            out_labels.append(EMPTY_LABEL)
        refs = torch.stack(refs, dim=0)
        mix = refs[:n_src].sum(dim=0)
        bg_path = row.get("background_path", "")
        if bg_path:
            bg = self._load_path(bg_path)
            snr_db = float(row.get("background_snr_db", "30.0"))
            scale = torch.sqrt(mix.pow(2).mean() / (10.0 ** (snr_db / 10.0)) / (bg.pow(2).mean() + 1e-8))
            mix = mix + bg * scale
        scale = self.amp_range / (mix.abs().max() + 1e-8)
        return {
            "mix": (mix * scale).float(),
            "ref": (refs * scale).float(),
            "labels": torch.tensor(out_labels).long(),
            "n_src": torch.tensor(n_src).long(),
        }


def make_single_event_dataloader(
    root: str,
    split: str,
    batch_size=64,
    num_workers=4,
    balanced=True,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    **kwargs,
):
    dataset = DASSingleEventDataset(root=root, split=split, **kwargs)
    sampler = None
    shuffle = split == "train"
    if split == "train" and balanced:
        labels = torch.tensor(dataset.labels)
        class_count = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
        weights = (1.0 / (class_count + 1e-8))[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        "drop_last": False,
        "persistent_workers": (num_workers > 0) if persistent_workers is None else persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def make_das_mix_dataloader(
    root: str,
    split: str,
    batch_size=4,
    num_workers=4,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    **kwargs,
):
    dataset = DASRandomMixDataset(root=root, split=split, **kwargs)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": (split == "train"),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        "drop_last": (split == "train"),
        "persistent_workers": (num_workers > 0) if persistent_workers is None else persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def make_das_manifest_dataloader(
    root: str,
    manifest_csv: str,
    batch_size=8,
    num_workers=4,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    **kwargs,
):
    dataset = DASManifestMixDataset(root=root, manifest_csv=manifest_csv, **kwargs)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else pin_memory,
        "drop_last": False,
        "persistent_workers": (num_workers > 0) if persistent_workers is None else persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)
