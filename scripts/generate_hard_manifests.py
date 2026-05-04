from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from das_sep.data import CLASS_NAMES, scan_class_files


FIELDS = [
    "scenario",
    "sample_index",
    "n_src",
    "source_paths",
    "labels",
    "gains",
    "shifts",
    "spatial_params",
    "background_path",
    "background_snr_db",
]


def rel_path(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def choose_n_src(rng: random.Random, scenario: str) -> int:
    if scenario in {"hard4src", "near_overlap"}:
        return 4
    if scenario == "sameclass":
        return 4 if rng.random() < 0.75 else 3
    if scenario == "low_snr":
        return 4 if rng.random() < 0.70 else 3
    r = rng.random()
    if r < 0.45:
        return 2
    if r < 0.80:
        return 3
    return 4


def choose_labels(rng: random.Random, scenario: str, n_src: int) -> list[int]:
    target_labels = list(range(1, len(CLASS_NAMES)))
    if scenario == "sameclass":
        label = rng.choice(target_labels)
        return [label for _ in range(n_src)]
    if scenario == "standard500" and rng.random() < 0.10:
        return rng.choices(target_labels, k=n_src)
    if n_src <= len(target_labels):
        return rng.sample(target_labels, n_src)
    return rng.choices(target_labels, k=n_src)


def choose_paths(rng: random.Random, class_to_files: dict[int, list[str]], labels: list[int]) -> list[str]:
    used: set[str] = set()
    paths: list[str] = []
    for label in labels:
        candidates = class_to_files[label]
        available = [path for path in candidates if path not in used]
        path = rng.choice(available or candidates)
        used.add(path)
        paths.append(path)
    return paths


def scenario_ranges(scenario: str) -> dict[str, object]:
    if scenario == "low_snr":
        return {
            "gain": (0.55, 1.55),
            "shift": 900,
            "bg_prob": 1.0,
            "bg_snr": (8.0, 16.0),
            "delay": 8,
            "slope": (-1.0, 1.0),
            "chan_gain": (0.86, 1.15),
            "width": (2.0, 5.5),
            "blend": (0.25, 0.75),
        }
    if scenario == "near_overlap":
        return {
            "gain": (0.65, 1.45),
            "shift": 160,
            "bg_prob": 0.25,
            "bg_snr": (22.0, 38.0),
            "delay": 12,
            "slope": (-1.4, 1.4),
            "chan_gain": (0.84, 1.18),
            "width": (1.8, 4.5),
            "blend": (0.35, 0.85),
        }
    if scenario == "sameclass":
        return {
            "gain": (0.68, 1.42),
            "shift": 650,
            "bg_prob": 0.15,
            "bg_snr": (24.0, 40.0),
            "delay": 8,
            "slope": (-0.9, 0.9),
            "chan_gain": (0.88, 1.12),
            "width": (2.2, 6.0),
            "blend": (0.20, 0.70),
        }
    if scenario == "hard4src":
        return {
            "gain": (0.62, 1.50),
            "shift": 700,
            "bg_prob": 0.20,
            "bg_snr": (22.0, 38.0),
            "delay": 8,
            "slope": (-1.0, 1.0),
            "chan_gain": (0.88, 1.12),
            "width": (2.0, 6.0),
            "blend": (0.20, 0.75),
        }
    return {
        "gain": (0.55, 1.65),
        "shift": 1200,
        "bg_prob": 0.60,
        "bg_snr": (14.0, 28.0),
        "delay": 0,
        "slope": (0.0, 0.0),
        "chan_gain": (1.0, 1.0),
        "width": (12.0, 12.0),
        "blend": (0.0, 0.0),
    }


def make_spatial_params(rng: random.Random, ranges: dict[str, object], channels: int = 12) -> dict[str, object]:
    delay = int(ranges["delay"])
    if delay > 0:
        slope_low, slope_high = ranges["slope"]
        slope = rng.uniform(slope_low, slope_high)
        offset = rng.uniform(-delay, delay)
        mid = (channels - 1) / 2.0
        shifts = [
            max(-delay, min(delay, int(round(offset + (ch - mid) * slope))))
            for ch in range(channels)
        ]
    else:
        shifts = [0 for _ in range(channels)]
    gain_low, gain_high = ranges["chan_gain"]
    width_low, width_high = ranges["width"]
    blend_low, blend_high = ranges["blend"]
    return {
        "channel_shifts": shifts,
        "per_channel_gains": [rng.uniform(gain_low, gain_high) for _ in range(channels)],
        "spatial_center": rng.uniform(0.0, float(channels - 1)),
        "spatial_width": rng.uniform(width_low, width_high),
        "spatial_blend": rng.uniform(blend_low, blend_high),
    }


def make_row(
    rng: random.Random,
    *,
    root: str,
    class_to_files: dict[int, list[str]],
    scenario: str,
    sample_index: int,
) -> dict[str, str]:
    n_src = choose_n_src(rng, scenario)
    labels = choose_labels(rng, scenario, n_src)
    paths = choose_paths(rng, class_to_files, labels)
    ranges = scenario_ranges(scenario)
    gain_low, gain_high = ranges["gain"]
    max_shift = int(ranges["shift"])
    bg_path = ""
    bg_snr = ""
    if rng.random() < float(ranges["bg_prob"]):
        bg_path = rel_path(root, rng.choice(class_to_files[0]))
        snr_low, snr_high = ranges["bg_snr"]
        bg_snr = f"{rng.uniform(snr_low, snr_high):.6f}"
    return {
        "scenario": scenario,
        "sample_index": str(sample_index),
        "n_src": str(n_src),
        "source_paths": json.dumps([rel_path(root, path) for path in paths], ensure_ascii=True),
        "labels": json.dumps(labels, ensure_ascii=True),
        "gains": json.dumps([rng.uniform(gain_low, gain_high) for _ in range(n_src)], ensure_ascii=True),
        "shifts": json.dumps([rng.randint(-max_shift, max_shift) for _ in range(n_src)], ensure_ascii=True),
        "spatial_params": json.dumps([make_spatial_params(rng, ranges) for _ in range(n_src)], ensure_ascii=True),
        "background_path": bg_path,
        "background_snr_db": bg_snr,
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/hy-tmp/DAS-dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_dir", default=str(ROOT / "manifests" / "v16_hard_tests"))
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument("--standard_samples", type=int, default=4000)
    parser.add_argument("--hard_samples", type=int, default=1024)
    args = parser.parse_args()

    class_to_files = scan_class_files(args.root, args.split)
    scenarios = {
        "standard500": args.standard_samples,
        "hard4src": args.hard_samples,
        "sameclass": args.hard_samples,
        "low_snr": args.hard_samples,
        "near_overlap": args.hard_samples,
    }
    out_dir = Path(args.out_dir)
    for offset, (scenario, count) in enumerate(scenarios.items()):
        rng = random.Random(args.seed + offset * 1009)
        rows = [
            make_row(rng, root=args.root, class_to_files=class_to_files, scenario=scenario, sample_index=i)
            for i in range(count)
        ]
        path = out_dir / f"{scenario}.csv"
        write_manifest(path, rows)
        print(f"wrote {path} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
