from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from das_sep.data import make_das_mix_dataloader
from das_sep.evaluation import evaluate_separated_classification, evaluate_separator
from das_sep.models import DASMCConvTasNet, DASResNetClassifier
from das_sep.utils import get_device, load_config, load_model_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_4090.yml"))
    parser.add_argument(
        "--separator_config",
        default="",
        help="Optional config used only for separator net_conf, useful when evaluating expanded models on a fixed test protocol.",
    )
    parser.add_argument("--separator_ckpt", default="")
    parser.add_argument("--classifier_ckpt", default="")
    parser.add_argument("--out_dir", default=str(ROOT / "results" / "metrics"))
    parser.add_argument("--max_batches", type=int, default=200)
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
    sep_ckpt = args.separator_ckpt or str(Path(sep_conf["train"]["checkpoint"]) / "best.pt")
    cls_ckpt = args.classifier_ckpt or str(Path(cls_conf["train"]["checkpoint"]) / "best.pt")
    load_model_state(separator, sep_ckpt, device, strict=args.strict)
    load_model_state(classifier, cls_ckpt, device, strict=args.strict)
    loader = make_das_mix_dataloader(split="test", deterministic=True, **opt["separator"]["datasets"])
    evaluate_separator(separator, loader, device, str(Path(args.out_dir) / "separator_metrics.csv"), args.max_batches, args.post_smooth_kernel)
    metrics = evaluate_separated_classification(separator, classifier, loader, device, args.out_dir, args.max_batches, args.post_smooth_kernel)
    print(metrics)


if __name__ == "__main__":
    main()
