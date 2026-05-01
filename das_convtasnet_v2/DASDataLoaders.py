import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from das_preprocess import load_das_mat, crop_or_pad, to_cnn_input, normalize_amp


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
    class_to_files = {}
    for label, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")
        files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith(".mat")
        ]
        if len(files) == 0:
            raise RuntimeError(f"No .mat files found in {cls_dir}")
        class_to_files[label] = sorted(files)
    return class_to_files


def random_shift_zero(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    """x: [C, T]. Random temporal shift with zero padding instead of circular roll."""
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
    noise_power = noise.pow(2).mean() + 1e-8
    scale = torch.sqrt(sig_power / (10.0 ** (snr_db / 10.0)) / noise_power)
    return x + noise * scale


def random_time_mask(x: torch.Tensor, max_width: int = 0, prob: float = 0.0) -> torch.Tensor:
    """Small time masking for classifier robustness; x is [C, T] or [1, T, C]."""
    if max_width <= 0 or random.random() > prob:
        return x
    if x.dim() == 2:
        t = x.shape[-1]
        width = random.randint(1, min(max_width, t))
        start = random.randint(0, t - width)
        x = x.clone()
        x[:, start:start + width] = 0
        return x
    if x.dim() == 3:
        t = x.shape[1]
        width = random.randint(1, min(max_width, t))
        start = random.randint(0, t - width)
        x = x.clone()
        x[:, start:start + width, :] = 0
        return x
    return x


class DASSingleEventDataset(Dataset):
    """用于训练/测试单事件分类器。"""

    def __init__(
        self,
        root,
        split="train",
        chunk_size=10000,
        use_diff=True,
        augment=False,
        aug_gain=(0.8, 1.25),
        aug_noise_prob=0.3,
        aug_snr_db=(18.0, 35.0),
        aug_shift=800,
        aug_time_mask_prob=0.2,
        aug_time_mask_width=600,
        smooth_kernel=1,
    ):
        self.root = root
        self.split = split
        self.chunk_size = chunk_size
        self.use_diff = use_diff
        self.augment = augment and split == "train"
        self.aug_gain = aug_gain
        self.aug_noise_prob = aug_noise_prob
        self.aug_snr_db = aug_snr_db
        self.aug_shift = aug_shift
        self.aug_time_mask_prob = aug_time_mask_prob
        self.aug_time_mask_width = aug_time_mask_width
        self.smooth_kernel = smooth_kernel
        self.items = []

        class_to_files = scan_class_files(root, split)
        for label, files in class_to_files.items():
            for path in files:
                self.items.append((path, label))

    def __len__(self):
        return len(self.items)

    @property
    def labels(self):
        return [label for _, label in self.items]

    def __getitem__(self, index):
        path, label = self.items[index]
        x = load_das_mat(
            path,
            use_diff=self.use_diff,
            norm_mode="minmax",
            smooth_kernel=self.smooth_kernel,
        )  # [12, T]
        x = crop_or_pad(x, self.chunk_size, random_crop=(self.split == "train"))

        if self.augment:
            x = random_shift_zero(x, self.aug_shift)
            if random.random() < self.aug_noise_prob:
                x = add_white_noise(x, self.aug_snr_db)
            gain = random.uniform(*self.aug_gain)
            x = x * gain

        x = to_cnn_input(x).squeeze(0)  # [1, T, 12]
        if self.augment:
            x = random_time_mask(x, self.aug_time_mask_width, self.aug_time_mask_prob)
        return {"x": x.float(), "label": torch.tensor(label).long(), "path": path}


class DASRandomMixDataset(Dataset):
    """
    用单事件样本动态构造2/3/4源混合样本，用于Conv-TasNet分离训练。

    返回:
        mix:    [12, T]
        ref:    [K, 12, T]
        labels: [K]
        n_src:  标量，真实源个数

    建议：分离训练默认不把background作为一个要分离的目标源，而是作为mix的加性噪声。
    这样更符合“事件分离 + 背景干扰鲁棒”的应用目标，也避免模型把背景噪声当成一个强制分离源。
    """

    def __init__(
        self,
        root,
        split="train",
        chunk_size=10000,
        min_sources=2,
        max_sources=3,
        amp_range=0.01,
        allow_background=False,
        deterministic=False,
        use_diff=True,
        epoch_length=None,
        two_source_prob=0.6,
        three_source_prob=0.4,
        same_class_prob=0.10,
        gain_range=(0.55, 1.65),
        max_source_shift=1200,
        polarity_flip_prob=0.1,
        add_background_noise=True,
        background_noise_prob=0.6,
        background_snr_db=(14.0, 28.0),
        add_white_noise_prob=0.15,
        white_noise_snr_db=(20.0, 35.0),
        smooth_kernel=1,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.chunk_size = chunk_size
        self.min_sources = min_sources
        self.max_sources = max_sources
        self.amp_range = amp_range
        self.allow_background = allow_background
        self.deterministic = deterministic
        self.use_diff = use_diff
        self.two_source_prob = two_source_prob
        self.three_source_prob = three_source_prob
        self.same_class_prob = same_class_prob
        self.gain_range = gain_range
        self.max_source_shift = max_source_shift
        self.polarity_flip_prob = polarity_flip_prob
        self.add_background_noise = add_background_noise
        self.background_noise_prob = background_noise_prob
        self.background_snr_db = background_snr_db
        self.add_white_noise_prob = add_white_noise_prob
        self.white_noise_snr_db = white_noise_snr_db
        self.smooth_kernel = smooth_kernel

        self.class_to_files = scan_class_files(root, split)
        default_length = sum(len(v) for v in self.class_to_files.values())
        # Larger virtual epoch gives many more random mixtures without copying data.
        self.epoch_length = epoch_length or (default_length * 2 if split == "train" else default_length)

    def __len__(self):
        return self.epoch_length

    def _choose_n_src(self):
        if self.min_sources == self.max_sources:
            return self.max_sources
        if self.min_sources == 2 and self.max_sources == 3:
            return 2 if random.random() < self.two_source_prob else 3
        if self.min_sources == 2 and self.max_sources == 4:
            r = random.random()
            if r < self.two_source_prob:
                return 2
            if r < self.two_source_prob + self.three_source_prob:
                return 3
            return 4
        return random.randint(self.min_sources, self.max_sources)

    def _load_signal(self, label):
        path = random.choice(self.class_to_files[label])
        sig = load_das_mat(
            path,
            use_diff=self.use_diff,
            norm_mode="amp",
            amp_range=self.amp_range,
            smooth_kernel=self.smooth_kernel,
        )
        sig = crop_or_pad(sig, self.chunk_size, random_crop=(self.split == "train"))
        return sig

    def _choose_labels(self, n_src):
        candidate_labels = list(range(len(CLASS_NAMES)))
        if not self.allow_background:
            candidate_labels = candidate_labels[1:]

        # Mostly different classes, sometimes same class to improve robustness.
        if random.random() < self.same_class_prob:
            return random.choices(candidate_labels, k=n_src)
        if n_src <= len(candidate_labels):
            return random.sample(candidate_labels, n_src)
        return random.choices(candidate_labels, k=n_src)

    def __getitem__(self, index):
        if self.deterministic:
            random.seed(index)
            torch.manual_seed(index)

        n_src = self._choose_n_src()
        labels = self._choose_labels(n_src)

        refs = []
        out_labels = []
        for lab in labels:
            sig = self._load_signal(lab)
            if self.split == "train":
                sig = random_shift_zero(sig, self.max_source_shift)
                gain = random.uniform(*self.gain_range)
                sig = sig * gain
                if random.random() < self.polarity_flip_prob:
                    sig = -sig
            refs.append(sig)
            out_labels.append(lab)

        while len(refs) < self.max_sources:
            refs.append(torch.zeros_like(refs[0]))
            out_labels.append(EMPTY_LABEL)

        refs = torch.stack(refs, dim=0)  # [K, 12, T]
        mix = refs[:n_src].sum(dim=0)    # [12, T]

        # Add background as noise, not as a source target.
        if (
            self.add_background_noise
            and len(self.class_to_files.get(0, [])) > 0
            and random.random() < self.background_noise_prob
        ):
            bg = self._load_signal(0)
            snr_db = random.uniform(*self.background_snr_db)
            sig_power = mix.pow(2).mean()
            bg_power = bg.pow(2).mean() + 1e-8
            scale = torch.sqrt(sig_power / (10.0 ** (snr_db / 10.0)) / bg_power)
            mix = mix + bg * scale

        if self.split == "train" and random.random() < self.add_white_noise_prob:
            mix = add_white_noise(mix, self.white_noise_snr_db)

        # 保证 mix = sum(refs)+noise 后总体幅值在合理范围；refs与mix使用同一scale。
        scale = self.amp_range / (mix.abs().max() + 1e-8)
        mix = mix * scale
        refs = refs * scale

        return {
            "mix": mix.float(),
            "ref": refs.float(),
            "labels": torch.tensor(out_labels).long(),
            "n_src": torch.tensor(n_src).long(),
        }


def make_single_event_dataloader(
    root,
    split,
    batch_size=64,
    num_workers=4,
    chunk_size=10000,
    use_diff=True,
    augment=False,
    balanced=True,
    **augment_kwargs,
):
    dataset = DASSingleEventDataset(
        root=root,
        split=split,
        chunk_size=chunk_size,
        use_diff=use_diff,
        augment=augment,
        **augment_kwargs,
    )

    sampler = None
    shuffle = split == "train"
    if split == "train" and balanced:
        labels = torch.tensor(dataset.labels)
        class_count = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
        weight_per_class = 1.0 / (class_count + 1e-8)
        sample_weights = weight_per_class[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def make_das_mix_dataloader(
    root,
    split,
    batch_size=4,
    num_workers=4,
    chunk_size=10000,
    min_sources=2,
    max_sources=3,
    amp_range=0.01,
    allow_background=False,
    deterministic=None,
    use_diff=True,
    epoch_length=None,
    two_source_prob=0.6,
    three_source_prob=0.4,
    same_class_prob=0.10,
    gain_range=(0.55, 1.65),
    max_source_shift=1200,
    polarity_flip_prob=0.1,
    add_background_noise=True,
    background_noise_prob=0.6,
    background_snr_db=(14.0, 28.0),
    add_white_noise_prob=0.15,
    white_noise_snr_db=(20.0, 35.0),
    smooth_kernel=1,
):
    if deterministic is None:
        deterministic = (split != "train")
    dataset = DASRandomMixDataset(
        root=root,
        split=split,
        chunk_size=chunk_size,
        min_sources=min_sources,
        max_sources=max_sources,
        amp_range=amp_range,
        allow_background=allow_background,
        deterministic=deterministic,
        use_diff=use_diff,
        epoch_length=epoch_length,
        two_source_prob=two_source_prob,
        three_source_prob=three_source_prob,
        same_class_prob=same_class_prob,
        gain_range=tuple(gain_range),
        max_source_shift=max_source_shift,
        polarity_flip_prob=polarity_flip_prob,
        add_background_noise=add_background_noise,
        background_noise_prob=background_noise_prob,
        background_snr_db=tuple(background_snr_db),
        add_white_noise_prob=add_white_noise_prob,
        white_noise_snr_db=tuple(white_noise_snr_db),
        smooth_kernel=smooth_kernel,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
