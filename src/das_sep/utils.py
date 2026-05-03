from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_seed(seed: int = 2026, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def ensure_dir(path: str | os.PathLike | None):
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def get_device(gpu_ids=None) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and (gpu_ids is None or len(gpu_ids) > 0) else "cpu")


def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def count_parameters(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def save_checkpoint(path: str, model, optimizer=None, scheduler=None, epoch: int = 0, extra: dict | None = None):
    ensure_dir(Path(path).parent)
    state = {"epoch": epoch, "model_state_dict": model.state_dict()}
    if optimizer is not None:
        state["optim_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_model_state(model, checkpoint_path: str, device: torch.device, strict: bool = True, ignore_mismatch: bool = False) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if ignore_mismatch:
        current = model.state_dict()
        filtered = {}
        skipped = {}
        for key, value in state.items():
            if key in current and current[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped[key] = tuple(value.shape) if hasattr(value, "shape") else "unknown"
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if strict:
            raise RuntimeError("ignore_mismatch requires strict=False")
        out = ckpt if isinstance(ckpt, dict) else {"model_state_dict": ckpt}
        out["skipped_mismatch"] = skipped
        out["missing_keys"] = missing
        out["unexpected_keys"] = unexpected
        return out
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}")
    return ckpt if isinstance(ckpt, dict) else {"model_state_dict": ckpt}
