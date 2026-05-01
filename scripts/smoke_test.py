from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from das_sep.data import make_das_mix_dataloader, make_single_event_dataloader
from das_sep.losses import pit_si_snr_variable_sources
from das_sep.models import DASMCConvTasNet, DASResNetClassifier
from das_sep.utils import get_device, load_config, setup_seed, to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke_local.yml"))
    args = parser.parse_args()
    opt = load_config(args.config)
    setup_seed(opt.get("seed", 2026), True)
    device = get_device(opt.get("gpu_ids", []))

    sep_conf = opt["separator"]
    cls_conf = opt["classifier"]
    mix_loader = make_das_mix_dataloader(split="train", **sep_conf["datasets"])
    cls_loader = make_single_event_dataloader(split="train", **cls_conf["datasets"])

    mix_batch = to_device(next(iter(mix_loader)), device)
    cls_batch = to_device(next(iter(cls_loader)), device)
    separator = DASMCConvTasNet(**sep_conf["net_conf"]).to(device)
    classifier = DASResNetClassifier(**cls_conf["net_conf"]).to(device)

    ests = separator(mix_batch["mix"])
    loss, sisnr, _ = pit_si_snr_variable_sources(ests, mix_batch["ref"], mix_batch["n_src"], mix=mix_batch["mix"], **sep_conf["train"].get("loss", {}))
    logits = classifier(cls_batch["x"])
    assert ests.shape[:3] == mix_batch["ref"].shape[:3], (ests.shape, mix_batch["ref"].shape)
    assert logits.shape[1] == 6, logits.shape
    assert torch.isfinite(loss), loss
    assert torch.isfinite(logits).all(), logits
    loss.backward()
    print("smoke_test_ok")
    print({"mix": tuple(mix_batch["mix"].shape), "ref": tuple(mix_batch["ref"].shape), "ests": tuple(ests.shape), "logits": tuple(logits.shape), "loss": float(loss.detach().cpu()), "si_snr": float(sisnr.detach().cpu())})


if __name__ == "__main__":
    main()
