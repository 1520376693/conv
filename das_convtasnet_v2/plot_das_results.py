import argparse
import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from utils import ensure_dir


def load_mat_data(path):
    mat = sio.loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise RuntimeError(f"No valid matrix in {path}")
    arr = mat[keys[0]].astype(np.float32)
    if arr.shape[0] >= arr.shape[1]:
        arr = arr.T  # [12, T]
    return arr


def plot_heatmap(x, title, save_path):
    plt.figure(figsize=(10, 4))
    plt.imshow(x, aspect="auto", origin="lower")
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time sample")
    plt.ylabel("Spatial channel")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def select_max_energy_channel(x):
    energy = np.mean(x ** 2, axis=1)
    ch = int(np.argmax(energy))
    return x[ch], ch


def plot_waveforms(signals, names, save_path):
    plt.figure(figsize=(12, 6))
    offset = 0.0
    for x, name in zip(signals, names):
        wave, ch = select_max_energy_channel(x)
        wave = wave / (np.max(np.abs(wave)) + 1e-8)
        plt.plot(wave + offset, linewidth=0.8, label=f"{name}, ch={ch+1}")
        offset += 2.0
    plt.xlabel("Time sample")
    plt.ylabel("Normalized amplitude + offset")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sum_check(signals, names, save_path):
    mix = None
    ests = []
    for x, name in zip(signals, names):
        if name == "mix":
            mix = x
        elif name.startswith("est_"):
            ests.append(x)
    if mix is None or not ests:
        return
    est_sum = np.sum(np.stack(ests, axis=0), axis=0)
    wave_mix, ch = select_max_energy_channel(mix)
    wave_sum = est_sum[ch]
    wave_mix = wave_mix / (np.max(np.abs(wave_mix)) + 1e-8)
    wave_sum = wave_sum / (np.max(np.abs(wave_sum)) + 1e-8)
    plt.figure(figsize=(12, 4))
    plt.plot(wave_mix, linewidth=0.8, label=f"mix, ch={ch+1}")
    plt.plot(wave_sum, linewidth=0.8, label=f"sum(est), ch={ch+1}")
    plt.xlabel("Time sample")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True) + 1e-8
    cm_norm = cm / row_sum

    plt.figure(figsize=(7, 6))
    plt.imshow(cm_norm, aspect="auto")
    plt.colorbar(label="Normalized count")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True, help="separate_das.py输出的某个样本文件夹")
    parser.add_argument("--save_dir", type=str, default="results/figures")
    args = parser.parse_args()

    ensure_dir(args.save_dir)
    mat_files = [f for f in os.listdir(args.sample_dir) if f.lower().endswith(".mat")]
    signals, names = [], []
    for name in sorted(mat_files):
        path = os.path.join(args.sample_dir, name)
        x = load_mat_data(path)
        signals.append(x)
        stem = os.path.splitext(name)[0]
        names.append(stem)
        plot_heatmap(x, name, os.path.join(args.save_dir, f"{stem}_heatmap.png"))

    plot_waveforms(signals, names, os.path.join(args.save_dir, "waveform_compare.png"))
    plot_sum_check(signals, names, os.path.join(args.save_dir, "mix_vs_sum_est.png"))
    print(f"Figures saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
