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


ROOT = Path(__file__).resolve().parents[1]


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
    values = [float(row[key]) for row in rows]
    return sum(values) / max(len(values), 1)


def parse_separator_metrics(out_dir: Path) -> dict[str, float]:
    csv_path = out_dir / "separator_metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if int(row.get("ref_idx", row.get("source_index", "-1"))) >= 0
        ]
    metrics = {
        "si_snr": mean(rows, "si_snr"),
        "si_snri": mean(rows, "si_snri"),
    }
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


def candidate_name(path: Path) -> str:
    return path.stem.replace(".", "_")


def find_candidates(checkpoint_dir: Path, include_best_last: bool) -> list[Path]:
    candidates = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if include_best_last:
        for name in ("best.pt", "last.pt"):
            path = checkpoint_dir / name
            if path.exists():
                candidates.append(path)
    seen = set()
    unique = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def score_metrics(metrics: dict[str, float], weights: tuple[float, float, float]) -> float:
    si_snri = metrics.get("si_snri", float("-inf"))
    si_snri_4src = metrics.get("si_snri_4src", float("-inf"))
    si_snr = metrics.get("si_snr", float("-inf"))
    if not all(math.isfinite(v) for v in (si_snri, si_snri_4src, si_snr)):
        return float("-inf")
    return weights[0] * si_snri + weights[1] * si_snri_4src + weights[2] * si_snr


def write_summary(path: Path, results: list[CandidateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CandidateResult.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.score, reverse=True):
            writer.writerow(result.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_config", default="configs/train_4090.yml")
    parser.add_argument("--separator_config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--classifier_ckpt", default="checkpoints/classifier_sep_finetune_hard34/best.pt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--summary_csv", default="")
    parser.add_argument("--max_batches", type=int, default=120)
    parser.add_argument("--post_smooth_kernel", type=int, default=1)
    parser.add_argument("--min_accuracy", type=float, default=0.90)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--weights", nargs=3, type=float, default=(0.55, 0.35, 0.10))
    parser.add_argument("--include_best_last", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = ROOT / args.checkpoint_dir
    out_root = ROOT / args.out_dir
    summary_csv = ROOT / args.summary_csv if args.summary_csv else out_root / "checkpoint_selection_summary.csv"
    candidates = find_candidates(checkpoint_dir, include_best_last=args.include_best_last)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint candidates found in {checkpoint_dir}")

    results: list[CandidateResult] = []
    for ckpt in candidates:
        name = candidate_name(ckpt)
        out_dir = out_root / name
        if not args.skip_existing or not (out_dir / "separator_metrics.csv").exists():
            run_command(
                [
                    sys.executable,
                    "scripts/evaluate_pipeline.py",
                    "--config",
                    args.standard_config,
                    "--separator_config",
                    args.separator_config,
                    "--separator_ckpt",
                    str(ckpt),
                    "--classifier_ckpt",
                    args.classifier_ckpt,
                    "--out_dir",
                    str(out_dir),
                    "--max_batches",
                    str(args.max_batches),
                    "--post_smooth_kernel",
                    str(args.post_smooth_kernel),
                ],
                out_root / f"evaluate_{name}.log",
            )
        metrics = parse_separator_metrics(out_dir)
        metrics.update(parse_classification_metrics(out_dir))
        score = score_metrics(metrics, tuple(args.weights))
        if metrics.get("accuracy", 0.0) < args.min_accuracy:
            score = float("-inf")
        results.append(
            CandidateResult(
                checkpoint=str(ckpt),
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
        )
        write_summary(summary_csv, results)

    valid = [item for item in results if math.isfinite(item.score)]
    if not valid:
        raise RuntimeError(f"No candidate met min_accuracy={args.min_accuracy:.4f}")
    selected = max(valid, key=lambda item: item.score)
    target = checkpoint_dir / "metric_best.pt"
    shutil.copy2(ROOT / selected.checkpoint, target)
    print(
        f"Selected {selected.checkpoint} -> {target} "
        f"score={selected.score:.6f} si_snri={selected.si_snri:.6f} "
        f"si_snr={selected.si_snr:.6f} acc={selected.accuracy:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
