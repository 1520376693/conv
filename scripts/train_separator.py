from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from das_sep.data import make_das_mix_dataloader
from das_sep.models import DASMCConvTasNet
from das_sep.trainers import SeparatorTrainer
from das_sep.utils import get_device, load_config, setup_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_4090.yml"))
    args = parser.parse_args()
    opt = load_config(args.config)
    setup_seed(opt.get("seed", 2026), opt.get("deterministic", False))
    gpu_ids = opt.get("gpu_ids", [0])
    if torch.cuda.is_available() and gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    device = get_device(gpu_ids)
    sep_conf = opt["separator"]
    model = DASMCConvTasNet(**sep_conf["net_conf"])
    ds = sep_conf["datasets"]
    train_loader = make_das_mix_dataloader(split="train", **ds)
    val_loader = make_das_mix_dataloader(split="test", **ds)
    SeparatorTrainer(model, sep_conf, device).fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
