from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import ensure_dir


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def plot_heatmap(x, title: str, save_path: str):
    x = _to_numpy(x)
    ensure_dir(Path(save_path).parent)
    plt.figure(figsize=(10, 4))
    plt.imshow(x, aspect="auto", origin="lower")
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time sample")
    plt.ylabel("Spatial channel")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _max_energy_wave(x):
    x = _to_numpy(x)
    ch = int(np.argmax(np.mean(x**2, axis=1)))
    w = x[ch]
    return w / (np.max(np.abs(w)) + 1e-8), ch


def plot_waveform_compare(signals: list, names: list, save_path: str):
    ensure_dir(Path(save_path).parent)
    plt.figure(figsize=(12, 6))
    offset = 0.0
    for sig, name in zip(signals, names):
        w, ch = _max_energy_wave(sig)
        plt.plot(w + offset, linewidth=0.8, label=f"{name}, ch={ch + 1}")
        offset += 2.0
    plt.xlabel("Time sample")
    plt.ylabel("Normalized amplitude + offset")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_qualitative_case(mix, refs, ests, best_info: dict, labels, save_dir: str):
    ensure_dir(save_dir)
    plot_heatmap(mix, "mix", os.path.join(save_dir, "mix_heatmap.png"))
    signals = [mix]
    names = ["mix"]
    n = len(best_info["perm"])
    for i in range(n):
        ref_idx = best_info["perm"][i]
        out_idx = best_info["out_ids"][i]
        label = int(labels[ref_idx].item()) if isinstance(labels, torch.Tensor) else int(labels[ref_idx])
        ref = refs[ref_idx]
        est = ests[out_idx]
        plot_heatmap(ref, f"ref_{i + 1}_label_{label}", os.path.join(save_dir, f"ref_{i + 1}_heatmap.png"))
        plot_heatmap(est, f"est_{i + 1}_label_{label}", os.path.join(save_dir, f"est_{i + 1}_heatmap.png"))
        signals.extend([ref, est])
        names.extend([f"ref_{i + 1}", f"est_{i + 1}"])
    plot_waveform_compare(signals, names, os.path.join(save_dir, "waveform_compare.png"))
    active_sum = torch.stack([ests[idx] for idx in best_info["out_ids"]], dim=0).sum(dim=0)
    plot_waveform_compare([mix, active_sum], ["mix", "sum_est"], os.path.join(save_dir, "mix_vs_sum_est.png"))
