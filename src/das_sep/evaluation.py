from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from .data import ID_TO_CLASS
from .losses import mae, mse, pearson_corr, pit_si_snr_variable_sources, sdr, si_snr_pair_single, snr
from .preprocess import moving_average_torch, to_cnn_input
from .utils import ensure_dir, to_device


def compute_separator_rows(mix, refs, ests, labels, n_src, best_infos, batch_idx=0):
    rows = []
    for b in range(mix.shape[0]):
        n = int(n_src[b].item())
        info = best_infos[b]
        active_sum = torch.zeros_like(mix[b])
        for i, ref_idx in enumerate(info["perm"]):
            out_idx = info["out_ids"][i]
            ref = refs[b, ref_idx]
            est = ests[b, out_idx]
            active_sum = active_sum + est
            si_snr_est = si_snr_pair_single(est, ref).item()
            si_snr_mix = si_snr_pair_single(mix[b], ref).item()
            label_id = int(labels[b, ref_idx].item()) if labels is not None else -1
            rows.append(
                {
                    "batch": batch_idx,
                    "sample": b,
                    "source_index": i,
                    "label_id": label_id,
                    "label_name": ID_TO_CLASS.get(label_id, "unknown"),
                    "n_src": n,
                    "out_idx": out_idx,
                    "ref_idx": ref_idx,
                    "si_snr": si_snr_est,
                    "si_snri": si_snr_est - si_snr_mix,
                    "snr": snr(est, ref).item(),
                    "sdr": sdr(est, ref).item(),
                    "mse": mse(est, ref).item(),
                    "mae": mae(est, ref).item(),
                    "pcc": pearson_corr(est, ref).item(),
                }
            )
        rows.append(
            {
                "batch": batch_idx,
                "sample": b,
                "source_index": -1,
                "label_id": -1,
                "label_name": "active_sum",
                "n_src": n,
                "out_idx": -1,
                "ref_idx": -1,
                "si_snr": si_snr_pair_single(active_sum, mix[b]).item(),
                "si_snri": 0.0,
                "snr": snr(active_sum, mix[b]).item(),
                "sdr": sdr(active_sum, mix[b]).item(),
                "mse": mse(active_sum, mix[b]).item(),
                "mae": mae(active_sum, mix[b]).item(),
                "pcc": pearson_corr(active_sum, mix[b]).item(),
            }
        )
    return rows


@torch.no_grad()
def evaluate_separator(separator, loader, device, out_csv: str, max_batches: int = 100, post_smooth_kernel: int = 1):
    separator.eval()
    rows = []
    for batch_idx, batch in enumerate(tqdm(loader, ncols=120, desc="eval_separator")):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        ests = separator(batch["mix"])
        if post_smooth_kernel > 1:
            ests = moving_average_torch(ests, post_smooth_kernel)
        _, _, best_infos = pit_si_snr_variable_sources(
            ests,
            batch["ref"],
            batch["n_src"],
            silence_weight=0.0,
            channel_weight=0.35,
            waveform_weight=0.0,
            stft_weight=0.0,
            mix=batch["mix"],
            mix_consistency_weight=0.0,
        )
        rows.extend(compute_separator_rows(batch["mix"], batch["ref"], ests, batch.get("labels"), batch["n_src"], best_infos, batch_idx))
    ensure_dir(Path(out_csv).parent)
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


@torch.no_grad()
def evaluate_separated_classification(separator, classifier, loader, device, out_dir: str, max_batches=100, post_smooth_kernel=1):
    separator.eval()
    classifier.eval()
    y_true, y_pred, n_srcs = [], [], []
    for batch_idx, batch in enumerate(tqdm(loader, ncols=120, desc="eval_sep_cls")):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = to_device(batch, device)
        ests = separator(batch["mix"])
        if post_smooth_kernel > 1:
            ests = moving_average_torch(ests, post_smooth_kernel)
        _, _, best_infos = pit_si_snr_variable_sources(
            ests,
            batch["ref"],
            batch["n_src"],
            silence_weight=0.0,
            channel_weight=0.35,
            waveform_weight=0.0,
            stft_weight=0.0,
            mix=batch["mix"],
            mix_consistency_weight=0.0,
        )
        for b, info in enumerate(best_infos):
            for i, ref_idx in enumerate(info["perm"]):
                out_idx = info["out_ids"][i]
                true = int(batch["labels"][b, ref_idx].item())
                if true >= 6:
                    continue
                logits = classifier(to_cnn_input(ests[b, out_idx]).to(device))
                y_true.append(true)
                y_pred.append(int(logits.argmax(dim=1).item()))
                n_srcs.append(int(batch["n_src"][b].item()))

    ensure_dir(out_dir)
    metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
        metrics = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
        np.savetxt(os.path.join(out_dir, "separated_classification_confusion.csv"), cm, fmt="%d", delimiter=",")
        report = classification_report(y_true, y_pred, labels=list(range(6)), target_names=[ID_TO_CLASS[i] for i in range(6)], zero_division=0)
        with open(os.path.join(out_dir, "separated_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n")
            for key, value in metrics.items():
                f.write(f"{key}={value:.6f}\n")
            for n in sorted(set(n_srcs)):
                ids = [i for i, v in enumerate(n_srcs) if v == n]
                f.write(f"accuracy_{n}src={accuracy_score([y_true[i] for i in ids], [y_pred[i] for i in ids]):.6f}\n")
    return metrics
