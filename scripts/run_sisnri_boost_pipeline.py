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
class Stage:
    name: str
    config: str
    checkpoint: str
    out_dir: str


STAGES = [
    Stage(
        name="v4a_r6_clean4src",
        config="configs/train_4090_sisnri_v4_r6_clean4src.yml",
        checkpoint="checkpoints/separator_sisnri_v4_r6_clean4src/best.pt",
        out_dir="results/final_metrics_sisnri_v4_r6_clean4src_standard_smooth1",
    ),
    Stage(
        name="v4b_r6_standard_calib",
        config="configs/train_4090_sisnri_v4_r6_standard_calib.yml",
        checkpoint="checkpoints/separator_sisnri_v4_r6_standard_calib/best.pt",
        out_dir="results/final_metrics_sisnri_v4_r6_standard_calib_smooth1",
    ),
    Stage(
        name="v5_r8_standard",
        config="configs/train_4090_sisnri_v5_r8_standard.yml",
        checkpoint="checkpoints/separator_sisnri_v5_r8_standard/best.pt",
        out_dir="results/final_metrics_sisnri_v5_r8_standard_smooth1",
    ),
]


def run_command(cmd: list[str], log_path: Path | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
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
        rows = [row for row in csv.DictReader(f) if int(row["source_index"]) != -1]
    metrics = {
        "si_snr": mean(rows, "si_snr"),
        "si_snri": mean(rows, "si_snri"),
        "snr": mean(rows, "snr"),
        "sdr": mean(rows, "sdr"),
    }
    for n_src in (2, 3, 4):
        group = [row for row in rows if int(row["n_src"]) == n_src]
        metrics[f"si_snr_{n_src}src"] = mean(group, "si_snr")
        metrics[f"si_snri_{n_src}src"] = mean(group, "si_snri")
    return metrics


def parse_classification_metrics(out_dir: Path) -> dict[str, float]:
    report = (out_dir / "separated_classification_report.txt").read_text(encoding="utf-8")
    metrics: dict[str, float] = {}
    for key in ["accuracy", "precision", "recall", "f1", "accuracy_2src", "accuracy_3src", "accuracy_4src"]:
        match = re.search(rf"^{key}=([0-9.]+)", report, flags=re.MULTILINE)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def evaluate(stage: Stage, standard_config: str, classifier_ckpt: str, max_batches: int, post_smooth_kernel: int) -> dict[str, float]:
    out_dir = ROOT / stage.out_dir
    run_command(
        [
            sys.executable,
            "scripts/evaluate_pipeline.py",
            "--config",
            standard_config,
            "--separator_config",
            stage.config,
            "--separator_ckpt",
            stage.checkpoint,
            "--classifier_ckpt",
            classifier_ckpt,
            "--out_dir",
            stage.out_dir,
            "--max_batches",
            str(max_batches),
            "--post_smooth_kernel",
            str(post_smooth_kernel),
        ],
        ROOT / f"evaluate_{stage.name}.log",
    )
    metrics = parse_separator_metrics(out_dir)
    metrics.update(parse_classification_metrics(out_dir))
    return metrics


def write_summary(results: list[tuple[str, dict[str, float]]]) -> None:
    summary_path = ROOT / "results" / "sisnri_boost_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "si_snr",
        "si_snri",
        "si_snr_2src",
        "si_snri_2src",
        "si_snr_3src",
        "si_snri_3src",
        "si_snr_4src",
        "si_snri_4src",
        "accuracy",
        "f1",
        "accuracy_2src",
        "accuracy_3src",
        "accuracy_4src",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stage, metrics in results:
            row = {"stage": stage}
            row.update({key: metrics.get(key, 0.0) for key in fieldnames if key != "stage"})
            writer.writerow(row)
    print(f"Wrote {summary_path}", flush=True)


def target_met(metrics: dict[str, float], target_si_snri: float, min_accuracy: float) -> bool:
    return metrics.get("si_snri", -1e9) >= target_si_snri and metrics.get("accuracy", 0.0) >= min_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_config", default="configs/train_4090.yml")
    parser.add_argument("--classifier_ckpt", default="checkpoints/classifier_sep_finetune_hard34/best.pt")
    parser.add_argument("--target_si_snri", type=float, default=5.0)
    parser.add_argument("--min_accuracy", type=float, default=0.90)
    parser.add_argument("--max_batches", type=int, default=500)
    parser.add_argument("--post_smooth_kernel", type=int, default=1)
    parser.add_argument("--skip_existing_train", action="store_true")
    args = parser.parse_args()

    results: list[tuple[str, dict[str, float]]] = []
    for stage in STAGES:
        ckpt_path = ROOT / stage.checkpoint
        if args.skip_existing_train and ckpt_path.exists():
            print(f"Skip training {stage.name}; found {ckpt_path}", flush=True)
        else:
            run_command(
                [sys.executable, "scripts/train_separator.py", "--config", stage.config],
                ROOT / f"train_separator_{stage.name}.log",
            )
        metrics = evaluate(stage, args.standard_config, args.classifier_ckpt, args.max_batches, args.post_smooth_kernel)
        results.append((stage.name, metrics))
        write_summary(results)
        print(
            f"{stage.name}: SI-SNR={metrics.get('si_snr', 0.0):.4f}, "
            f"SI-SNRi={metrics.get('si_snri', 0.0):.4f}, "
            f"accuracy={metrics.get('accuracy', 0.0):.4f}, "
            f"4src_SI-SNRi={metrics.get('si_snri_4src', 0.0):.4f}",
            flush=True,
        )
        if target_met(metrics, args.target_si_snri, args.min_accuracy):
            print(f"Target met after {stage.name}; stopping.", flush=True)
            break
        print(f"Target not met after {stage.name}; continuing if another stage exists.", flush=True)


if __name__ == "__main__":
    main()
