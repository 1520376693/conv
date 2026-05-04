from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from das_sep.data import make_das_manifest_dataloader
from das_sep.evaluation import evaluate_separated_classification, evaluate_separator
from das_sep.models import DASMCConvTasNet, DASResNetClassifier
from das_sep.utils import get_device, load_config, load_model_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_4090.yml"))
    parser.add_argument("--separator_config", default="")
    parser.add_argument("--separator_ckpt", required=True)
    parser.add_argument("--classifier_ckpt", default="checkpoints/classifier_sep_finetune_hard34/best.pt")
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--post_smooth_kernel", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    opt = load_config(args.config)
    sep_model_opt = load_config(args.separator_config) if args.separator_config else opt
    device = get_device(opt.get("gpu_ids", [0]))
    sep_conf = sep_model_opt["separator"]
    cls_conf = opt["classifier"]

    separator = DASMCConvTasNet(**sep_conf["net_conf"]).to(device)
    classifier = DASResNetClassifier(**cls_conf["net_conf"]).to(device)
    load_model_state(separator, args.separator_ckpt, device, strict=args.strict)
    load_model_state(classifier, args.classifier_ckpt, device, strict=args.strict)

    ds_conf = opt["separator"]["datasets"]
    loader = make_das_manifest_dataloader(
        root=ds_conf["root"],
        manifest_csv=args.manifest_csv,
        batch_size=ds_conf.get("batch_size", 8),
        num_workers=ds_conf.get("num_workers", 4),
        pin_memory=ds_conf.get("pin_memory"),
        persistent_workers=ds_conf.get("persistent_workers"),
        prefetch_factor=ds_conf.get("prefetch_factor"),
        chunk_size=ds_conf.get("chunk_size", 10000),
        max_sources=ds_conf.get("max_sources", 4),
        amp_range=ds_conf.get("amp_range", 0.01),
        use_diff=ds_conf.get("use_diff", True),
        smooth_kernel=ds_conf.get("smooth_kernel", 1),
    )
    out_dir = Path(args.out_dir)
    evaluate_separator(
        separator,
        loader,
        device,
        str(out_dir / "separator_metrics.csv"),
        args.max_batches,
        args.post_smooth_kernel,
    )
    metrics = evaluate_separated_classification(
        separator,
        classifier,
        loader,
        device,
        str(out_dir),
        args.max_batches,
        args.post_smooth_kernel,
    )
    print(metrics)


if __name__ == "__main__":
    main()
