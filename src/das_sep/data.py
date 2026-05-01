from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    if shift == 0:
        return x
    y = torch.zeros_like(x)
    if shift > 0:
        y[:, shift:] = x[:, :-shift]
    else:
        y[:, :shift] = x[:, -shift:]
    return y


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
        self.class_to_files = scan_class_files(root, split)
        default_length = sum(len(v) for v in self.class_to_files.values())
        self.epoch_length = epoch_length or (default_length * 2 if split == "train" else default_length)

    def __len__(self):
        return self.epoch_length

    def _choose_n_src(self) -> int:
        if self.min_sources == self.max_sources:
            return self.max_sources
        if self.min_sources == 2 and self.max_sources >= 4:
            r = random.random()
            if r < self.two_source_prob:
                return 2
            if r < self.two_source_prob + self.three_source_prob:
                return 3
            return 4
        if self.min_sources == 2 and self.max_sources == 3:
            return 2 if random.random() < self.two_source_prob else 3
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

    def _choose_labels(self, n_src: int) -> List[int]:
        labels = list(range(len(CLASS_NAMES)))
        if not self.allow_background:
            labels = labels[1:]
        if random.random() < self.same_class_prob:
            return random.choices(labels, k=n_src)
        if n_src <= len(labels):
            return random.sample(labels, n_src)
        return random.choices(labels, k=n_src)

    def __getitem__(self, index):
        if self.deterministic:
            random.seed(index)
            torch.manual_seed(index)
        n_src = self._choose_n_src()
        labels = self._choose_labels(n_src)
        refs, out_labels = [], []
        for label in labels:
            sig = self._load_signal(label)
            if self.split == "train":
                sig = random_shift_zero(sig, self.max_source_shift)
                sig = sig * random.uniform(*self.gain_range)
                if random.random() < self.polarity_flip_prob:
                    sig = -sig
            refs.append(sig)
            out_labels.append(label)
        while len(refs) < self.max_sources:
            refs.append(torch.zeros_like(refs[0]))
            out_labels.append(EMPTY_LABEL)
        refs = torch.stack(refs, dim=0)
        mix = refs[:n_src].sum(dim=0)
        if self.add_background_noise and random.random() < self.background_noise_prob:
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


def make_single_event_dataloader(root: str, split: str, batch_size=64, num_workers=4, balanced=True, **kwargs):
    dataset = DASSingleEventDataset(root=root, split=split, **kwargs)
    sampler = None
    shuffle = split == "train"
    if split == "train" and balanced:
        labels = torch.tensor(dataset.labels)
        class_count = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
        weights = (1.0 / (class_count + 1e-8))[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def make_das_mix_dataloader(root: str, split: str, batch_size=4, num_workers=4, **kwargs):
    dataset = DASRandomMixDataset(root=root, split=split, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
