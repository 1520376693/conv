from __future__ import annotations

import argparse
from pathlib import Path

from run_night_aggressive_optimization import (
    Candidate,
    append_summary,
    final_eval,
    maybe_train,
    select_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


CANDIDATES = [
    Candidate(
        name="v11_v7_micro_polish",
        route="micro_polish",
        config="configs/train_4090_sisnri_v11_v7_micro_polish.yml",
        checkpoint_dir="checkpoints/separator_sisnri_v11_v7_micro_polish",
    ),
    Candidate(
        name="v12_standard_recover",
        route="standard_recover",
        config="configs/train_4090_sisnri_v12_standard_recover.yml",
        checkpoint_dir="checkpoints/separator_sisnri_v12_standard_recover",
    ),
    Candidate(
        name="v13_clean4src_polish",
        route="clean4src_polish",
        config="configs/train_4090_sisnri_v13_clean4src_polish.yml",
        checkpoint_dir="checkpoints/separator_sisnri_v13_clean4src_polish",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard_config", default="configs/train_4090.yml")
    parser.add_argument("--classifier_ckpt", default="checkpoints/classifier_sep_finetune_hard34/best.pt")
    parser.add_argument("--select_batches", type=int, default=120)
    parser.add_argument("--final_batches", type=int, default=500)
    parser.add_argument("--post_smooth_kernel", type=int, default=1)
    parser.add_argument("--min_accuracy", type=float, default=0.90)
    parser.add_argument("--summary_csv", default="results/polish_recovery_summary.csv")
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


if __name__ == "__main__":
    main()
