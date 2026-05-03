from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    name: str
    route: str
    config: str
    checkpoint_dir: str


CANDIDATES = [
    Candidate(
        name="v8_metric_select",
        route="hard4src_metric_select",
        config="configs/train_4090_sisnri_v8_metric_select.yml",
        checkpoint_dir="checkpoints/separator_sisnri_v8_metric_select",
    ),
    Candidate(
        name="v9_loss_light_consistency",
        route="loss_sweep",
        config="configs/train_4090_sisnri_v9_loss_light_consistency.yml",
        checkpoint_dir="checkpoints/separator_sisnri_v9_loss_light_consistency",
    ),
    Candidate(
        name="v9_loss_stft_micro",
        route="loss_sweep",
        config="configs/train_4090_sisnri_v9_loss_stft_micro.yml",
        checkpoint_dir="checkpoints/separator_sisnri_v9_loss_stft_micro",
    ),
    Candidate(
        name="v10_capacity_r10",
        route="capacity",
        config="configs/train_4090_sisnri_v10_capacity_r10.yml",
        checkpoint_dir="checkpoints/separator_sisnri_v10_capacity_r10",
    ),
]


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
    "accepted_accuracy",
    "beats_v7_si_snri",
    "beats_v7_si_snr",
]


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
    with (out_dir / "separator_metrics.csv").open(newline="", encoding="utf-8") as f:
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
    report = (out_dir / "separated_classification_report.txt").read_text(encoding="utf-8")
    metrics: dict[str, float] = {}
    for key in ["accuracy", "f1", "accuracy_2src", "accuracy_3src", "accuracy_4src"]:
        match = re.search(rf"^{key}=([0-9.]+)", report, flags=re.MULTILINE)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def append_summary(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def maybe_train(candidate: Candidate, skip_existing_train: bool) -> None:
    metric_best = ROOT / candidate.checkpoint_dir / "metric_best.pt"
    if skip_existing_train and metric_best.exists():
        print(f"Skip {candidate.name}; found {metric_best}", flush=True)
        return
    run_command(
        [sys.executable, "scripts/train_separator.py", "--config", candidate.config],
        ROOT / f"train_separator_{candidate.name}.log",
    )


def select_checkpoint(candidate: Candidate, args: argparse.Namespace) -> Path:
    out_dir = f"results/aggressive_select_{candidate.name}_smooth{args.post_smooth_kernel}"
    run_command(
        [
            sys.executable,
            "scripts/select_separator_checkpoint.py",
            "--standard_config",
            args.standard_config,
            "--separator_config",
            candidate.config,
            "--checkpoint_dir",
            candidate.checkpoint_dir,
            "--classifier_ckpt",
            args.classifier_ckpt,
            "--out_dir",
            out_dir,
            "--summary_csv",
            f"{out_dir}/checkpoint_selection_summary.csv",
            "--max_batches",
            str(args.select_batches),
            "--post_smooth_kernel",
            str(args.post_smooth_kernel),
            "--min_accuracy",
            str(args.min_accuracy),
            "--include_best_last",
        ],
        ROOT / f"select_checkpoint_{candidate.name}.log",
    )
    return ROOT / candidate.checkpoint_dir / "metric_best.pt"


def final_eval(candidate: Candidate, ckpt: Path, args: argparse.Namespace) -> dict[str, float]:
    out_dir = ROOT / f"results/aggressive_metrics_{candidate.name}_smooth{args.post_smooth_kernel}"
    run_command(
        [
            sys.executable,
            "scripts/evaluate_pipeline.py",
            "--config",
            args.standard_config,
            "--separator_config",
            candidate.config,
            "--separator_ckpt",
            str(ckpt),
            "--classifier_ckpt",
            args.classifier_ckpt,
            "--out_dir",
            str(out_dir),
            "--max_batches",
            str(args.final_batches),
            "--post_smooth_kernel",
            str(args.post_smooth_kernel),
        ],
        ROOT / f"evaluate_{candidate.name}_final.log",
    )
    metrics = parse_separator_metrics(out_dir)
    metrics.update(parse_classification_metrics(out_dir))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_config", default="configs/train_4090.yml")
    parser.add_argument("--classifier_ckpt", default="checkpoints/classifier_sep_finetune_hard34/best.pt")
    parser.add_argument("--select_batches", type=int, default=120)
    parser.add_argument("--final_batches", type=int, default=500)
    parser.add_argument("--post_smooth_kernel", type=int, default=1)
    parser.add_argument("--min_accuracy", type=float, default=0.90)
    parser.add_argument("--summary_csv", default="results/aggressive_optimization_summary.csv")
    parser.add_argument("--skip_existing_train", action="store_true")
    args = parser.parse_args()

    summary_path = ROOT / args.summary_csv
    for candidate in CANDIDATES:
        try:
            maybe_train(candidate, args.skip_existing_train)
            ckpt = select_checkpoint(candidate, args)
            metrics = final_eval(candidate, ckpt, args)
            row = {
                "stage": candidate.name,
                "route": candidate.route,
                "checkpoint": str(ckpt),
                **metrics,
                "accepted_accuracy": metrics.get("accuracy", 0.0) >= args.min_accuracy,
                "beats_v7_si_snri": metrics.get("si_snri", -1e9) > 3.614011,
                "beats_v7_si_snr": metrics.get("si_snr", -1e9) > 0.750797,
            }
            append_summary(summary_path, row)
            print(
                f"{candidate.name}: SI-SNR={metrics.get('si_snr', 0.0):.6f}, "
                f"SI-SNRi={metrics.get('si_snri', 0.0):.6f}, "
                f"4src={metrics.get('si_snri_4src', 0.0):.6f}, "
                f"acc={metrics.get('accuracy', 0.0):.6f}",
                flush=True,
            )
        except Exception as exc:
            append_summary(
                summary_path,
                {
                    "stage": candidate.name,
                    "route": candidate.route,
                    "checkpoint": "FAILED",
                    "accepted_accuracy": False,
                    "beats_v7_si_snri": False,
                    "beats_v7_si_snr": False,
                },
            )
            print(f"{candidate.name} failed: {exc}", flush=True)
            continue


if __name__ == "__main__":
    main()
