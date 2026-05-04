from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

V14_SI_SNR = 0.759940
V14_SI_SNRI = 3.623154
V14_4SRC_SI_SNRI = 3.716309

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
    "main_model_candidate",
]

HARD_FIELDS = [
    "model",
    "scenario",
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
    values = (
        metrics.get("si_snri", float("-inf")),
        metrics.get("si_snri_4src", float("-inf")),
        metrics.get("si_snr", float("-inf")),
    )
    if not all(math.isfinite(value) for value in values):
        return float("-inf")
    return weights[0] * values[0] + weights[1] * values[1] + weights[2] * values[2]


def append_csv(path: Path, fields: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def accepted_main(metrics: dict[str, float]) -> bool:
    return (
        metrics.get("accuracy", 0.0) >= 0.90
        and metrics.get("si_snri", float("-inf")) > V14_SI_SNRI
        and metrics.get("si_snr", float("-inf")) >= V14_SI_SNR
        and metrics.get("si_snri_4src", float("-inf")) >= V14_4SRC_SI_SNRI
    )


def cleanup_epoch_checkpoints(route_names: list[str]) -> None:
    freed = 0
    removed = 0
    for route in route_names:
        ckpt_dir = ROOT / "checkpoints" / f"separator_sisnri_{route}"
        if not ckpt_dir.exists():
            continue
        for path in ckpt_dir.glob("epoch_*.pt"):
            try:
                size = path.stat().st_size
                path.unlink()
                freed += size
                removed += 1
            except FileNotFoundError:
                pass
    print(f"cleanup_epoch_checkpoints removed={removed} freed_gb={freed / 1024**3:.2f}", flush=True)


def ensure_disk_space(min_free_gb: float) -> None:
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    print(f"disk_free_gb={free_gb:.2f}", flush=True)
    if free_gb >= min_free_gb:
        return
    cleanup_epoch_checkpoints(
        [
            "v8_metric_select",
            "v9_loss_light_consistency",
            "v9_loss_stft_micro",
            "v10_capacity_r10",
            "v11_v7_micro_polish",
            "v12_standard_recover",
            "v15_sourcebalance_polish",
            "v16a_realistic_mix_continue",
            "v16b_realistic_hard4_curriculum",
            "v17_spatial_stem2d",
        ]
    )
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    print(f"disk_free_gb_after_cleanup={free_gb:.2f}", flush=True)


def wait_for_pid(pid_file: str, poll_seconds: int) -> None:
    path = ROOT / pid_file
    if not path.exists():
        print(f"wait_for_pid skipped; missing {path}", flush=True)
        return
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        print(f"wait_for_pid skipped; invalid pid file {path}", flush=True)
        return
    while True:
        proc = subprocess.run(["bash", "-lc", f"ps -p {pid} -o pid="], cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            print(f"wait_for_pid finished; pid {pid} is not active", flush=True)
            return
        print(f"waiting_for_existing_v17_pid={pid}", flush=True)
        time.sleep(poll_seconds)


def checkpoint_candidates(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("epoch_*.pt")) + [
        path for path in [checkpoint_dir / "best.pt", checkpoint_dir / "last.pt"] if path.exists()
    ]


def smoke_stage(stage: str, config: str, checkpoint_dir: str, args: argparse.Namespace) -> None:
    done_marker = ROOT / "results" / "_smoke_v18" / f"{stage}.done"
    if done_marker.exists() or (ROOT / checkpoint_dir / "metric_best.pt").exists():
        return
    with (ROOT / config).open(encoding="utf-8") as f:
        smoke_conf = yaml.safe_load(f)
    smoke_conf["name"] = f"{smoke_conf.get('name', stage)}_smoke"
    ds = smoke_conf["separator"]["datasets"]
    ds["epoch_length"] = 512
    ds["val_epoch_length"] = 128
    train = smoke_conf["separator"]["train"]
    train["checkpoint"] = f"./checkpoints/_smoke_{stage}"
    train["num_epochs"] = 1
    train["early_stop"] = 1
    train["save_epoch_checkpoints"] = False
    smoke_config = ROOT / "results" / "_smoke_v18" / f"{stage}.yml"
    smoke_config.parent.mkdir(parents=True, exist_ok=True)
    if (ROOT / "checkpoints" / f"_smoke_{stage}").exists():
        shutil.rmtree(ROOT / "checkpoints" / f"_smoke_{stage}")
    with smoke_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(smoke_conf, f, sort_keys=False)
    run_command(
        [sys.executable, "scripts/train_separator.py", "--config", str(smoke_config)],
        ROOT / "logs" / "v18_followup" / f"smoke_{stage}.log",
    )
    done_marker.write_text("ok\n", encoding="utf-8")


def train_select_stage(stage: str, config: str, checkpoint_dir: str, args: argparse.Namespace) -> Path:
    ckpt_dir = ROOT / checkpoint_dir
    metric_best = ckpt_dir / "metric_best.pt"
    if not metric_best.exists():
        smoke_stage(stage, config, checkpoint_dir, args)
    if not metric_best.exists() and not checkpoint_candidates(ckpt_dir):
        run_command(
            [sys.executable, "scripts/train_separator.py", "--config", config],
            ROOT / "logs" / "v18_followup" / f"train_{stage}.log",
        )
    if not metric_best.exists():
        run_command(
            [
                sys.executable,
                "scripts/select_separator_checkpoint.py",
                "--standard_config",
                args.standard_config,
                "--separator_config",
                config,
                "--checkpoint_dir",
                checkpoint_dir,
                "--classifier_ckpt",
                args.classifier_ckpt,
                "--out_dir",
                f"results/{stage}_checkpoint_select",
                "--summary_csv",
                f"results/{stage}_checkpoint_select/checkpoint_selection_summary.csv",
                "--max_batches",
                str(args.final_batches),
                "--post_smooth_kernel",
                str(args.post_smooth_kernel),
                "--min_accuracy",
                "0.90",
                "--include_best_last",
                "--skip_existing",
            ],
            ROOT / "logs" / "v18_followup" / f"select_{stage}.log",
        )
    if not metric_best.exists():
        raise FileNotFoundError(f"metric_best was not created for {stage}: {metric_best}")
    return metric_best


def evaluate_standard(
    *,
    model: str,
    separator_config: str,
    checkpoint: Path,
    out_dir: Path,
    args: argparse.Namespace,
    weights: tuple[float, float, float],
) -> dict[str, float]:
    if not (out_dir / "separator_metrics.csv").exists():
        run_command(
            [
                sys.executable,
                "scripts/evaluate_pipeline.py",
                "--config",
                args.standard_config,
                "--separator_config",
                separator_config,
                "--separator_ckpt",
                str(checkpoint),
                "--classifier_ckpt",
                args.classifier_ckpt,
                "--out_dir",
                str(out_dir),
                "--max_batches",
                str(args.final_batches),
                "--post_smooth_kernel",
                str(args.post_smooth_kernel),
            ],
            ROOT / "logs" / "v18_followup" / f"evaluate_standard_{model}.log",
        )
    metrics = parse_separator_metrics(out_dir)
    metrics.update(parse_classification_metrics(out_dir))
    metrics["score"] = score_metrics(metrics, weights)
    append_csv(
        ROOT / args.summary_csv,
        SUMMARY_FIELDS,
        {
            "stage": model,
            "route": "standard",
            "checkpoint": str(checkpoint),
            **metrics,
            "accepted_accuracy": metrics.get("accuracy", 0.0) >= 0.90,
            "beats_v14_si_snri": metrics.get("si_snri", 0.0) > V14_SI_SNRI,
            "beats_v14_si_snr": metrics.get("si_snr", 0.0) >= V14_SI_SNR,
            "beats_v14_4src_si_snri": metrics.get("si_snri_4src", 0.0) >= V14_4SRC_SI_SNRI,
            "main_model_candidate": accepted_main(metrics),
        },
    )
    return metrics


def evaluate_hard_suite(
    *,
    model: str,
    separator_config: str,
    checkpoint: Path,
    args: argparse.Namespace,
    weights: tuple[float, float, float],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    manifest_dir = ROOT / args.manifest_dir
    for scenario in ["standard500", "hard4src", "sameclass", "low_snr", "near_overlap"]:
        out_dir = ROOT / "results" / "v18_hard_eval" / model / scenario
        if not (out_dir / "separator_metrics.csv").exists():
            run_command(
                [
                    sys.executable,
                    "scripts/evaluate_manifest.py",
                    "--config",
                    args.standard_config,
                    "--separator_config",
                    separator_config,
                    "--separator_ckpt",
                    str(checkpoint),
                    "--classifier_ckpt",
                    args.classifier_ckpt,
                    "--manifest_csv",
                    str(manifest_dir / f"{scenario}.csv"),
                    "--out_dir",
                    str(out_dir),
                    "--max_batches",
                    "0",
                    "--post_smooth_kernel",
                    str(args.post_smooth_kernel),
                ],
                ROOT / "logs" / "v18_followup" / f"evaluate_manifest_{model}_{scenario}.log",
            )
        metrics = parse_separator_metrics(out_dir)
        metrics.update(parse_classification_metrics(out_dir))
        metrics["score"] = score_metrics(metrics, weights)
        append_csv(
            ROOT / args.hard_summary_csv,
            HARD_FIELDS,
            {"model": model, "scenario": scenario, "checkpoint": str(checkpoint), **metrics},
        )
        results[scenario] = metrics
    return results


def run_stage(
    stage: str,
    config: str,
    checkpoint_dir: str,
    args: argparse.Namespace,
    weights: tuple[float, float, float],
) -> dict[str, float]:
    ensure_disk_space(args.min_free_gb)
    checkpoint = train_select_stage(stage, config, checkpoint_dir, args)
    standard = evaluate_standard(
        model=stage,
        separator_config=config,
        checkpoint=checkpoint,
        out_dir=ROOT / "results" / "v18_standard_eval" / stage,
        args=args,
        weights=weights,
    )
    evaluate_hard_suite(model=stage, separator_config=config, checkpoint=checkpoint, args=args, weights=weights)
    return standard


def ensure_manifests(args: argparse.Namespace) -> None:
    manifest_dir = ROOT / args.manifest_dir
    expected = [manifest_dir / f"{name}.csv" for name in ["standard500", "hard4src", "sameclass", "low_snr", "near_overlap"]]
    if all(path.exists() for path in expected):
        return
    run_command(
        [
            sys.executable,
            "scripts/generate_hard_manifests.py",
            "--root",
            args.dataset_root,
            "--out_dir",
            str(manifest_dir),
            "--standard_samples",
            str(args.standard_manifest_samples),
            "--hard_samples",
            str(args.hard_manifest_samples),
        ],
        ROOT / "logs" / "v18_followup" / "generate_hard_manifests.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_config", default="configs/train_4090.yml")
    parser.add_argument("--classifier_ckpt", default="checkpoints/classifier_sep_finetune_hard34/best.pt")
    parser.add_argument("--dataset_root", default="/hy-tmp/DAS-dataset")
    parser.add_argument("--manifest_dir", default="manifests/v16_hard_tests")
    parser.add_argument("--standard_manifest_samples", type=int, default=4000)
    parser.add_argument("--hard_manifest_samples", type=int, default=1024)
    parser.add_argument("--final_batches", type=int, default=500)
    parser.add_argument("--post_smooth_kernel", type=int, default=1)
    parser.add_argument("--summary_csv", default="results/v18_followup_summary.csv")
    parser.add_argument("--hard_summary_csv", default="results/v18_followup_hard_summary.csv")
    parser.add_argument("--wait_for_pid_file", default="logs/run_v16_v17_optimization.pid")
    parser.add_argument("--wait_poll_seconds", type=int, default=300)
    parser.add_argument("--min_free_gb", type=float, default=8.0)
    args = parser.parse_args()
    weights = (0.55, 0.35, 0.10)

    if args.wait_for_pid_file:
        wait_for_pid(args.wait_for_pid_file, args.wait_poll_seconds)
    ensure_manifests(args)
    ensure_disk_space(args.min_free_gb)

    v17_standard = run_stage(
        "v17_spatial_stem2d",
        "configs/train_4090_sisnri_v17_spatial_stem2d.yml",
        "checkpoints/separator_sisnri_v17_spatial_stem2d",
        args,
        weights,
    )
    if accepted_main(v17_standard):
        print("v17 beats v14 under the fixed standard protocol; v18 will not run.", flush=True)
        return
    cleanup_epoch_checkpoints(["v17_spatial_stem2d"])

    v18a_standard = run_stage(
        "v18a_standard_hard_balance",
        "configs/train_4090_sisnri_v18a_standard_hard_balance.yml",
        "checkpoints/separator_sisnri_v18a_standard_hard_balance",
        args,
        weights,
    )
    if accepted_main(v18a_standard):
        print("v18A beats v14 under the fixed standard protocol; v18B will not run.", flush=True)
        return

    v18b_standard = run_stage(
        "v18b_multiscale_encoder",
        "configs/train_4090_sisnri_v18b_multiscale_encoder.yml",
        "checkpoints/separator_sisnri_v18b_multiscale_encoder",
        args,
        weights,
    )
    best = max([v17_standard, v18a_standard, v18b_standard], key=lambda item: item.get("score", float("-inf")))
    print(
        "v18 follow-up finished. Best new standard score "
        f"SI-SNR={best.get('si_snr', 0.0):.6f} "
        f"SI-SNRi={best.get('si_snri', 0.0):.6f} "
        f"4src={best.get('si_snri_4src', 0.0):.6f} "
        f"acc={best.get('accuracy', 0.0):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
