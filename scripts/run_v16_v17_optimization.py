from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path


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
            ROOT / "logs" / "v16_v17" / f"evaluate_standard_{model}.log",
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
    scenarios = ["standard500", "hard4src", "sameclass", "low_snr", "near_overlap"]
    for scenario in scenarios:
        out_dir = ROOT / "results" / "v16_hard_eval" / model / scenario
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
                ROOT / "logs" / "v16_v17" / f"evaluate_manifest_{model}_{scenario}.log",
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


def hard_score(results: dict[str, dict[str, float]]) -> float:
    scenarios = ["hard4src", "sameclass", "low_snr", "near_overlap"]
    return sum(results[name].get("si_snri", 0.0) for name in scenarios) / len(scenarios)


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
        ROOT / "logs" / "v16_v17" / "generate_hard_manifests.log",
    )


def checkpoint_candidates(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("epoch_*.pt")) + [
        path for path in [checkpoint_dir / "best.pt", checkpoint_dir / "last.pt"] if path.exists()
    ]


def train_select_stage(stage: str, config: str, checkpoint_dir: str, args: argparse.Namespace) -> Path:
    ckpt_dir = ROOT / checkpoint_dir
    metric_best = ckpt_dir / "metric_best.pt"
    if not metric_best.exists() and not checkpoint_candidates(ckpt_dir):
        run_command(
            [sys.executable, "scripts/train_separator.py", "--config", config],
            ROOT / "logs" / "v16_v17" / f"train_{stage}.log",
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
            ROOT / "logs" / "v16_v17" / f"select_{stage}.log",
        )
    if not metric_best.exists():
        raise FileNotFoundError(f"metric_best was not created for {stage}: {metric_best}")
    return metric_best


def run_stage(
    stage: str,
    config: str,
    checkpoint_dir: str,
    args: argparse.Namespace,
    weights: tuple[float, float, float],
) -> tuple[dict[str, float], dict[str, dict[str, float]], Path]:
    ckpt = train_select_stage(stage, config, checkpoint_dir, args)
    standard = evaluate_standard(
        model=stage,
        separator_config=config,
        checkpoint=ckpt,
        out_dir=ROOT / "results" / "v16_standard_eval" / stage,
        args=args,
        weights=weights,
    )
    hard = evaluate_hard_suite(
        model=stage,
        separator_config=config,
        checkpoint=ckpt,
        args=args,
        weights=weights,
    )
    return standard, hard, ckpt


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
    parser.add_argument("--summary_csv", default="results/v16_v17_optimization_summary.csv")
    parser.add_argument("--hard_summary_csv", default="results/v16_v17_hard_summary.csv")
    parser.add_argument("--hard_improve_delta", type=float, default=0.005)
    args = parser.parse_args()
    weights = (0.55, 0.35, 0.10)

    ensure_manifests(args)
    v14_ckpt = ROOT / "checkpoints/separator_sisnri_v14_v13_micro_polish/metric_best.pt"
    v14_hard = evaluate_hard_suite(
        model="v14_v13_micro_polish",
        separator_config="configs/train_4090_sisnri_v14_v13_micro_polish.yml",
        checkpoint=v14_ckpt,
        args=args,
        weights=weights,
    )

    v16a_standard, v16a_hard, _ = run_stage(
        "v16a_realistic_mix_continue",
        "configs/train_4090_sisnri_v16a_realistic_mix_continue.yml",
        "checkpoints/separator_sisnri_v16a_realistic_mix_continue",
        args,
        weights,
    )
    best_standard = v16a_standard
    if accepted_main(v16a_standard):
        print("v16a beats v14 under the fixed standard protocol.", flush=True)
        return

    ran_v16b = False
    if hard_score(v16a_hard) > hard_score(v14_hard) + args.hard_improve_delta:
        ran_v16b = True
        v16b_standard, _, _ = run_stage(
            "v16b_realistic_hard4_curriculum",
            "configs/train_4090_sisnri_v16b_realistic_hard4_curriculum.yml",
            "checkpoints/separator_sisnri_v16b_realistic_hard4_curriculum",
            args,
            weights,
        )
        best_standard = max([best_standard, v16b_standard], key=lambda item: item.get("score", float("-inf")))
        if accepted_main(v16b_standard):
            print("v16b beats v14 under the fixed standard protocol.", flush=True)
            return

    print(
        "v16 did not produce a standard-protocol replacement; "
        f"v16b_ran={ran_v16b}. Starting v17 spatial stem 2D.",
        flush=True,
    )
    v17_standard, _, _ = run_stage(
        "v17_spatial_stem2d",
        "configs/train_4090_sisnri_v17_spatial_stem2d.yml",
        "checkpoints/separator_sisnri_v17_spatial_stem2d",
        args,
        weights,
    )
    best_standard = max([best_standard, v17_standard], key=lambda item: item.get("score", float("-inf")))
    print(
        "v16/v17 finished. Best new standard score "
        f"SI-SNR={best_standard.get('si_snr', 0.0):.6f} "
        f"SI-SNRi={best_standard.get('si_snri', 0.0):.6f} "
        f"4src={best_standard.get('si_snri_4src', 0.0):.6f} "
        f"acc={best_standard.get('accuracy', 0.0):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
