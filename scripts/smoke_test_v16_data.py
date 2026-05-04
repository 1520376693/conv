from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from das_sep.data import DASManifestMixDataset, DASRandomMixDataset, scan_class_files


def assert_sample(sample: dict, max_sources: int = 4) -> None:
    assert tuple(sample["mix"].shape) == (12, 10000), sample["mix"].shape
    assert tuple(sample["ref"].shape) == (max_sources, 12, 10000), sample["ref"].shape
    assert tuple(sample["labels"].shape) == (max_sources,), sample["labels"].shape
    assert sample["n_src"].item() in {2, 3, 4}, sample["n_src"]
    assert torch.isfinite(sample["mix"]).all()
    assert torch.isfinite(sample["ref"]).all()
    assert sample["mix"].abs().max().item() <= 0.0101


def write_tiny_manifest(root: str, path: Path) -> None:
    files = scan_class_files(root, "test")
    root_path = Path(root)
    row = {
        "scenario": "smoke",
        "sample_index": "0",
        "n_src": "2",
        "source_paths": json.dumps(
            [
                str(Path(files[1][0]).relative_to(root_path)).replace("\\", "/"),
                str(Path(files[2][0]).relative_to(root_path)).replace("\\", "/"),
            ]
        ),
        "labels": json.dumps([1, 2]),
        "gains": json.dumps([1.0, 0.85]),
        "shifts": json.dumps([0, 120]),
        "spatial_params": json.dumps(
            [
                {
                    "channel_shifts": [0] * 12,
                    "per_channel_gains": [1.0] * 12,
                    "spatial_center": 5.5,
                    "spatial_width": 12.0,
                    "spatial_blend": 0.0,
                },
                {
                    "channel_shifts": [0, 1, 1, 2, 2, 2, 1, 1, 0, -1, -1, -2],
                    "per_channel_gains": [0.95 + 0.01 * i for i in range(12)],
                    "spatial_center": 4.5,
                    "spatial_width": 3.2,
                    "spatial_blend": 0.45,
                },
            ]
        ),
        "background_path": str(Path(files[0][0]).relative_to(root_path)).replace("\\", "/"),
        "background_snr_db": "28.0",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/hy-tmp/DAS-dataset")
    parser.add_argument("--manifest_csv", default="")
    args = parser.parse_args()

    random_ds = DASRandomMixDataset(
        root=args.root,
        split="train",
        epoch_length=4,
        spatial_mix_prob=1.0,
        channel_delay_max=8,
        channel_delay_slope_range=(-1.0, 1.0),
        per_channel_gain_range=(0.88, 1.12),
        spatial_center_width_range=(2.0, 6.0),
        spatial_blend_range=(0.25, 0.75),
    )
    assert_sample(random_ds[0])

    manifest_csv = Path(args.manifest_csv) if args.manifest_csv else ROOT / "results" / "_smoke_v16_manifest.csv"
    if not manifest_csv.exists():
        write_tiny_manifest(args.root, manifest_csv)
    manifest_ds = DASManifestMixDataset(root=args.root, manifest_csv=str(manifest_csv))
    first = manifest_ds[0]
    second = manifest_ds[0]
    assert_sample(first)
    assert torch.equal(first["mix"], second["mix"])
    assert torch.equal(first["ref"], second["ref"])
    assert torch.equal(first["labels"], second["labels"])
    assert torch.equal(first["n_src"], second["n_src"])
    print("v16 data smoke test passed", flush=True)


if __name__ == "__main__":
    main()
