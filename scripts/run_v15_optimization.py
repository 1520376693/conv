from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

V14_SI_SNR = 0.759940
V14_SI_SNRI = 3.623154
V14_4SRC_SI_SNRI = 3.716309
V14_ACCURACY = 0.981497

SUMMARY_FIELDS = [
    "stage",
    "route",
    "checkpoint",
    "si_snr",
    "si_snri",
    "si_snri_2src",
    "si_snri_3src",
    "si_snri_4src",
    "accuracy",
    "f1",
    "accuracy_2src",
    "accuracy_3src",
    "accuracy_4src",
    "score",
    "accepted_accuracy",
    "beats_v14_si_snri",
    "beats_v14_si_snr",
    "beats_v14_4src_si_snri",
]


@dataclass(frozen=True)
class CandidateResult:
    checkpoint: str
    out_dir: str
    score: float
    si_snr: float
    si_snri: float
    si_snri_2src: float
    si_snri_3src: float
    si_snri_4src: float
    accuracy: float
    f1: float
    accuracy_2src: float
    accuracy_3src: float
    accuracy_4src: float


def run_command(cmd: list[str], log_path: Path) -> None:
    print("$ " + " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def parse_separator_metrics(out_dir: Path) -> dict[str, float]:
    csv_path = out_dir / "separator_metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if int(row.get("ref_idx", row.get("source_index", "-1"))) >= 0
        ]
    metrics = {"si_snr": mean(rows, "si_snr"), "si_snri": mean(rows, "si_snri")}
    for n_src in (2, 3, 4):
        group = [row for row in rows if int(row["n_src"]) == n_src]
        metrics[f"si_snri_{n_src}src"] = mean(group, "si_snri")
    return metrics


def parse_classification_metrics(out_dir: Path) -> dict[str, float]:
    report_path = out_dir / "separated_classification_report.txt"
    metrics: dict[str, float] = {}
    if not report_path.exists():
        return metrics
    report = report_path.read_text(encoding="utf-8")
    for key in ["accuracy", "f1", "accuracy_2src", "accuracy_3src", "accuracy_4src"]:
        match = re.search(rf"^{key}=([0-9.]+)", report, flags=re.MULTILINE)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def score_metrics(metrics: dict[str, float], weights: tuple[float, float, float]) -> float:
    si_snri = metrics.get("si_snri", float("-inf"))
    si_snri_4src = metrics.get("si_snri_4src", float("-inf"))
    si_snr = metrics.get("si_snr", float("-inf"))
    if not all(math.isfinite(v) for v in (si_snri, si_snri_4src, si_snr)):
        return float("-inf")
    return weights[0] * si_snri + weights[1] * si_snri_4src + weights[2] * si_snr


def append_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def write_selection_summary(path: Path, results: list[CandidateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CandidateResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.score, reverse=True):
            writer.writerow(result.__dict__)


def candidate_label(path: Path) -> str:
    return path.stem.replace(".", "_")


def unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    separator_config: str,
    out_dir: Path,
    standard_config: str,
    classifier_ckpt: str,
    max_batches: int,
    post_smooth_kernel: int,
    min_accuracy: float,
    weights: tuple[float, float, float],
    log_dir: Path,
) -> CandidateResult:
    if not (out_dir / "separator_metrics.csv").exists():
        run_command(
            [
                sys.executable,
                "scripts/evaluate_pipeline.py",
                "--config",
                standard_config,
                "--separator_config",
                separator_config,
                "--separator_ckpt",
                str(checkpoint),
                "--classifier_ckpt",
                classifier_ckpt,
                "--out_dir",
                str(out_dir),
                "--max_batches",
                str(max_batches),
                "--post_smooth_kernel",
                str(post_smooth_kernel),
            ],
            log_dir / f"evaluate_{candidate_label(checkpoint)}.log",
        )
    metrics = parse_separator_metrics(out_dir)
    metrics.update(parse_classification_metrics(out_dir))
    score = score_metrics(metrics, weights)
    if metrics.get("accuracy", 0.0) < min_accuracy:
        score = float("-inf")
    return CandidateResult(
        checkpoint=str(checkpoint),
        out_dir=str(out_dir),
        score=score,
        si_snr=metrics.get("si_snr", 0.0),
        si_snri=metrics.get("si_snri", 0.0),
        si_snri_2src=metrics.get("si_snri_2src", 0.0),
        si_snri_3src=metrics.get("si_snri_3src", 0.0),
        si_snri_4src=metrics.get("si_snri_4src", 0.0),
        accuracy=metrics.get("accuracy", 0.0),
        f1=metrics.get("f1", 0.0),
        accuracy_2src=metrics.get("accuracy_2src", 0.0),
        accuracy_3src=metrics.get("accuracy_3src", 0.0),
        accuracy_4src=metrics.get("accuracy_4src", 0.0),
    )


def summarize_stage(summary_csv: Path, stage: str, route: str, result: CandidateResult) -> None:
    append_summary(
        summary_csv,
        {
            "stage": stage,
            "route": route,
            "checkpoint": result.checkpoint,
            "si_snr": result.si_snr,
            "si_snri": result.si_snri,
            "si_snri_2src": result.si_snri_2src,
            "si_snri_3src": result.si_snri_3src,
            "si_snri_4src": result.si_snri_4src,
            "accuracy": result.accuracy,
            "f1": result.f1,
            "accuracy_2src": result.accuracy_2src,
            "accuracy_3src": result.accuracy_3src,
            "accuracy_4src": result.accuracy_4src,
            "score": result.score,
            "accepted_accuracy": result.accuracy >= 0.90,
            "beats_v14_si_snri": result.si_snri > V14_SI_SNRI,
            "beats_v14_si_snr": result.si_snr >= V14_SI_SNR,
            "beats_v14_4src_si_snri": result.si_snri_4src >= V14_4SRC_SI_SNRI,
        },
    )


def copy_checkpoint(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / "metric_best.pt"
    shutil.copy2(src, target)
    shutil.copy2(src, dst_dir / "best.pt")
    shutil.copy2(src, dst_dir / "last.pt")
    return target


def run_v15a(args: argparse.Namespace, weights: tuple[float, float, float], summary_csv: Path) -> list[CandidateResult]:
    v14_dir = ROOT / "checkpoints/separator_sisnri_v14_v13_micro_polish"
    candidates = unique_existing(
        sorted(v14_dir.glob("epoch_*.pt"))
        + [v14_dir / "best.pt", v14_dir / "last.pt", v14_dir / "metric_best.pt"]
    )
    out_root = ROOT / "results/v15a_v14_full500_select"
    log_dir = ROOT / "logs/v15a_v14_full500_select"
    results: list[CandidateResult] = []
    for checkpoint in candidates:
        result = evaluate_checkpoint(
            checkpoint=checkpoint,
            separator_config="configs/train_4090_sisnri_v14_v13_micro_polish.yml",
            out_dir=out_root / candidate_label(checkpoint),
            standard_config=args.standard_config,
            classifier_ckpt=args.classifier_ckpt,
            max_batches=args.final_batches,
            post_smooth_kernel=args.post_smooth_kernel,
            min_accuracy=args.min_accuracy,
            weights=weights,
            log_dir=log_dir,
        )
        results.append(result)
        write_selection_summary(out_root / "checkpoint_selection_summary.csv", results)
    valid = [item for item in results if math.isfinite(item.score)]
    if not valid:
        raise RuntimeError("v15a found no checkpoint meeting min accuracy")
    selected = max(valid, key=lambda item: item.score)
    copy_checkpoint(ROOT / selected.checkpoint, ROOT / "checkpoints/separator_sisnri_v15a_v14_full500_select")
    summarize_stage(summary_csv, "v15a_v14_full500_select", "full500_select", selected)
    print(
        f"v15a selected {selected.checkpoint} "
        f"SI-SNR={selected.si_snr:.6f} SI-SNRi={selected.si_snri:.6f} "
        f"4src={selected.si_snri_4src:.6f} acc={selected.accuracy:.6f}",
        flush=True,
    )
    return sorted(valid, key=lambda item: item.score, reverse=True)


def average_checkpoints(paths: list[Path], target: Path) -> None:
    checkpoints = [torch.load(path, map_location="cpu") for path in paths]
    states = [ckpt["model_state_dict"] for ckpt in checkpoints]
    averaged = {}
    for key, first_value in states[0].items():
        if torch.is_tensor(first_value) and first_value.is_floating_point():
            averaged[key] = torch.stack([state[key].float() for state in states], dim=0).mean(dim=0).to(first_value.dtype)
        else:
            averaged[key] = first_value
    out = dict(checkpoints[0])
    out["model_state_dict"] = averaged
    out["epoch"] = int(checkpoints[0].get("epoch", 0))
    out["best_val"] = float(checkpoints[0].get("best_val", 0.0))
    out["best_sisnr"] = float(checkpoints[0].get("best_sisnr", out["best_val"]))
    out.pop("optim_state_dict", None)
    out.pop("scheduler_state_dict", None)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, target)


def run_v15b(
    args: argparse.Namespace,
    weights: tuple[float, float, float],
    summary_csv: Path,
    ranked_v15a: list[CandidateResult],
) -> CandidateResult:
    ckpt_dir = ROOT / "checkpoints/separator_sisnri_v15b_v14_swa"
    top2 = [ROOT / item.checkpoint for item in ranked_v15a[:2]]
    top3 = [ROOT / item.checkpoint for item in ranked_v15a[:3]]
    average_checkpoints(top2, ckpt_dir / "swa_top2.pt")
    if len(top3) >= 3:
        average_checkpoints(top3, ckpt_dir / "swa_top3.pt")
    candidates = unique_existing([ckpt_dir / "swa_top2.pt", ckpt_dir / "swa_top3.pt"])
    out_root = ROOT / "results/v15b_v14_swa"
    log_dir = ROOT / "logs/v15b_v14_swa"
    results: list[CandidateResult] = []
    for checkpoint in candidates:
        result = evaluate_checkpoint(
            checkpoint=checkpoint,
            separator_config="configs/train_4090_sisnri_v14_v13_micro_polish.yml",
            out_dir=out_root / candidate_label(checkpoint),
            standard_config=args.standard_config,
            classifier_ckpt=args.classifier_ckpt,
            max_batches=args.final_batches,
            post_smooth_kernel=args.post_smooth_kernel,
            min_accuracy=args.min_accuracy,
            weights=weights,
            log_dir=log_dir,
        )
        results.append(result)
        write_selection_summary(out_root / "checkpoint_selection_summary.csv", results)
    valid = [item for item in results if math.isfinite(item.score)]
    if not valid:
        raise RuntimeError("v15b found no averaged checkpoint meeting min accuracy")
    selected = max(valid, key=lambda item: item.score)
    shutil.copy2(ROOT / selected.checkpoint, ckpt_dir / "metric_best.pt")
    shutil.copy2(ROOT / selected.checkpoint, ckpt_dir / "best.pt")
    shutil.copy2(ROOT / selected.checkpoint, ckpt_dir / "last.pt")
    summarize_stage(summary_csv, "v15b_v14_swa", "checkpoint_average", selected)
    print(
        f"v15b selected {selected.checkpoint} "
        f"SI-SNR={selected.si_snr:.6f} SI-SNRi={selected.si_snri:.6f} "
        f"4src={selected.si_snri_4src:.6f} acc={selected.accuracy:.6f}",
        flush=True,
    )
    return selected


def write_v15_config(path: Path, batch_size: int, checkpoint: str, num_epochs: int, epoch_length: int, val_epoch_length: int) -> None:
    text = f"""name: DAS_ConvTasNet_4090_sisnri_v15_sourcebalance_polish
seed: 2051
deterministic: false
gpu_ids: [0]

separator:
  datasets:
    root: /hy-tmp/DAS-dataset
    batch_size: {batch_size}
    num_workers: 12
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 4
    chunk_size: 10000
    min_sources: 2
    max_sources: 4
    two_source_prob: 0.07
    three_source_prob: 0.17
    amp_range: 0.01
    allow_background: false
    epoch_length: {epoch_length}
    val_epoch_length: {val_epoch_length}
    use_diff: true
    smooth_kernel: 1
    same_class_prob: 0.07
    gain_range: [0.68, 1.42]
    max_source_shift: 850
    polarity_flip_prob: 0.03
    add_background_noise: true
    background_noise_prob: 0.12
    background_snr_db: [24.0, 40.0]
    add_white_noise_prob: 0.01
    white_noise_snr_db: [32.0, 46.0]
  net_conf:
    in_channels: 12
    out_channels: 12
    N: 256
    L: 20
    B: 256
    H: 512
    P: 3
    X: 8
    R: 8
    norm: gln
    max_sources: 4
    activate: sigmoid
    dropout: 0.035
    spatial_stem: true
    spatial_depth: 2
    mixture_consistency: false
  train:
    checkpoint: {checkpoint}
    init_checkpoint: ./checkpoints/separator_sisnri_v14_v13_micro_polish/metric_best.pt
    strict_init: true
    monitor: sisnr
    scheduler_on: sisnr
    save_epoch_checkpoints: true
    epoch_checkpoint_every: 1
    optimizer: adamw
    scheduler: plateau
    num_epochs: {num_epochs}
    clip_norm: 5
    min_lr: 3.0e-8
    patience: 2
    factor: 0.5
    early_stop: 4
    use_amp: true
    loss:
      silence_weight: 0.020
      channel_weight: 0.020
      waveform_weight: 0.006
      stft_weight: 0.0
      mix_consistency_weight: 0.006
    optimizer_kwargs:
      lr: 1.0e-7
      weight_decay: 4.0e-5
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def probe_batch_size(batch_size: int) -> bool:
    config = ROOT / f"configs/train_4090_sisnri_v15_probe_bs{batch_size}.yml"
    checkpoint = f"./checkpoints/_probe_v15_bs{batch_size}"
    write_v15_config(config, batch_size, checkpoint, num_epochs=1, epoch_length=512, val_epoch_length=128)
    try:
        run_command(
            [sys.executable, "scripts/train_separator.py", "--config", str(config.relative_to(ROOT))],
            ROOT / "logs/v15_batch_probe" / f"probe_bs{batch_size}.log",
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Batch probe bs={batch_size} failed: {exc}", flush=True)
        return False
    finally:
        shutil.rmtree(ROOT / checkpoint.replace("./", ""), ignore_errors=True)


def choose_batch_size() -> int:
    selected = 20
    for batch_size in (24, 28):
        if probe_batch_size(batch_size):
            selected = batch_size
        else:
            break
    print(f"Selected v15c batch_size={selected}", flush=True)
    return selected


def select_trained_v15c(
    args: argparse.Namespace,
    weights: tuple[float, float, float],
    summary_csv: Path,
) -> CandidateResult:
    checkpoint_dir = ROOT / "checkpoints/separator_sisnri_v15_sourcebalance_polish"
    candidates = unique_existing(sorted(checkpoint_dir.glob("epoch_*.pt")) + [checkpoint_dir / "best.pt", checkpoint_dir / "last.pt"])
    out_root = ROOT / "results/v15c_sourcebalance_polish_select"
    log_dir = ROOT / "logs/v15c_sourcebalance_polish_select"
    results: list[CandidateResult] = []
    for checkpoint in candidates:
        result = evaluate_checkpoint(
            checkpoint=checkpoint,
            separator_config="configs/train_4090_sisnri_v15_sourcebalance_polish.yml",
            out_dir=out_root / candidate_label(checkpoint),
            standard_config=args.standard_config,
            classifier_ckpt=args.classifier_ckpt,
            max_batches=args.final_batches,
            post_smooth_kernel=args.post_smooth_kernel,
            min_accuracy=args.min_accuracy,
            weights=weights,
            log_dir=log_dir,
        )
        results.append(result)
        write_selection_summary(out_root / "checkpoint_selection_summary.csv", results)
    valid = [item for item in results if math.isfinite(item.score)]
    if not valid:
        raise RuntimeError("v15c found no checkpoint meeting min accuracy")
    selected = max(valid, key=lambda item: item.score)
    shutil.copy2(ROOT / selected.checkpoint, checkpoint_dir / "metric_best.pt")
    summarize_stage(summary_csv, "v15c_sourcebalance_polish", "sourcebalance_polish", selected)
    print(
        f"v15c selected {selected.checkpoint} "
        f"SI-SNR={selected.si_snr:.6f} SI-SNRi={selected.si_snri:.6f} "
        f"4src={selected.si_snri_4src:.6f} acc={selected.accuracy:.6f}",
        flush=True,
    )
    return selected


def cleanup_v14_epochs() -> None:
    v14_dir = ROOT / "checkpoints/separator_sisnri_v14_v13_micro_polish"
    for path in v14_dir.glob("epoch_*.pt"):
        print(f"Removing v14 intermediate checkpoint {path}", flush=True)
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_config", default="configs/train_4090.yml")
    parser.add_argument("--classifier_ckpt", default="checkpoints/classifier_sep_finetune_hard34/best.pt")
    parser.add_argument("--final_batches", type=int, default=500)
    parser.add_argument("--post_smooth_kernel", type=int, default=1)
    parser.add_argument("--min_accuracy", type=float, default=0.90)
    parser.add_argument("--summary_csv", default="results/v15_optimization_summary.csv")
    parser.add_argument("--skip_v15a", action="store_true")
    parser.add_argument("--skip_v15b", action="store_true")
    parser.add_argument("--skip_v15c", action="store_true")
    parser.add_argument("--skip_batch_probe", action="store_true")
    parser.add_argument("--keep_v14_epochs", action="store_true")
    args = parser.parse_args()

    weights = (0.55, 0.35, 0.10)
    summary_csv = ROOT / args.summary_csv
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if summary_csv.exists():
        archive = summary_csv.with_suffix(".previous.csv")
        shutil.copy2(summary_csv, archive)
        summary_csv.unlink()

    ranked_v15a: list[CandidateResult] = []
    if not args.skip_v15a:
        ranked_v15a = run_v15a(args, weights, summary_csv)

    if not args.skip_v15b:
        if not ranked_v15a:
            raise RuntimeError("v15b requires v15a ranked checkpoints")
        run_v15b(args, weights, summary_csv, ranked_v15a)
        if not args.keep_v14_epochs:
            cleanup_v14_epochs()

    if not args.skip_v15c:
        batch_size = 20 if args.skip_batch_probe else choose_batch_size()
        write_v15_config(
            ROOT / "configs/train_4090_sisnri_v15_sourcebalance_polish.yml",
            batch_size,
            "./checkpoints/separator_sisnri_v15_sourcebalance_polish",
            num_epochs=8,
            epoch_length=18000,
            val_epoch_length=5000,
        )
        run_command(
            [sys.executable, "scripts/train_separator.py", "--config", "configs/train_4090_sisnri_v15_sourcebalance_polish.yml"],
            ROOT / "logs/v15c_sourcebalance_polish_train.log",
        )
        select_trained_v15c(args, weights, summary_csv)

    print(f"Wrote {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
