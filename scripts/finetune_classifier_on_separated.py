from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from das_sep.data import make_das_mix_dataloader
from das_sep.losses import pit_si_snr_variable_sources
from das_sep.models import DASMCConvTasNet, DASResNetClassifier
from das_sep.preprocess import to_cnn_input
from das_sep.utils import ensure_dir, get_device, get_logger, load_config, load_model_state, save_checkpoint, setup_seed, to_device


def collect_separated_batch(separator, batch, device, include_negative: bool = True):
    with torch.no_grad():
        ests = separator(batch["mix"])
        _, _, best_infos = pit_si_snr_variable_sources(
            ests,
            batch["ref"],
            batch["n_src"],
            silence_weight=0.0,
            channel_weight=0.35,
            waveform_weight=0.0,
            stft_weight=0.0,
            mix=batch["mix"],
            mix_consistency_weight=0.0,
        )
    xs, ys = [], []
    for b, info in enumerate(best_infos):
        for i, ref_idx in enumerate(info["perm"]):
            label = int(batch["labels"][b, ref_idx].item())
            if label >= 6:
                continue
            x = ests[b, info["out_ids"][i]].detach()
            xs.append(x)
            ys.append(label)
            if include_negative:
                xs.append(-x)
                ys.append(label)
    if not xs:
        return None, None
    return to_cnn_input(torch.stack(xs, dim=0)).to(device), torch.tensor(ys, device=device).long()


def run_epoch(separator, classifier, loader, optimizer, device, training: bool, include_negative: bool):
    classifier.train(training)
    separator.eval()
    losses, y_true, y_pred = [], [], []
    desc = "train_sep_cls" if training else "valid_sep_cls"
    for batch in tqdm(loader, ncols=120, desc=desc):
        batch = to_device(batch, device)
        x, y = collect_separated_batch(separator, batch, device, include_negative=include_negative and training)
        if x is None:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = classifier(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.03 if training else 0.0)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0) if y_true else (0, 0, 0, None)
    return float(np.mean(losses)) if losses else 0.0, acc, p, r, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_4090.yml"))
    parser.add_argument("--separator_ckpt", default="checkpoints/separator_4090/best.pt")
    parser.add_argument("--classifier_ckpt", default="checkpoints/classifier_4090/best.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints/classifier_sep_finetune")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train_epoch_length", type=int, default=12000)
    parser.add_argument("--val_epoch_length", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--train_min_sources", type=int, default=0)
    parser.add_argument("--train_max_sources", type=int, default=0)
    parser.add_argument("--train_two_source_prob", type=float, default=-1.0)
    parser.add_argument("--train_three_source_prob", type=float, default=-1.0)
    parser.add_argument("--train_same_class_prob", type=float, default=0.0)
    args = parser.parse_args()

    opt = load_config(args.config)
    setup_seed(opt.get("seed", 2026) + 17, opt.get("deterministic", False))
    gpu_ids = opt.get("gpu_ids", [0])
    if torch.cuda.is_available() and gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    device = get_device(gpu_ids)
    sep_conf = opt["separator"]
    cls_conf = opt["classifier"]

    separator = DASMCConvTasNet(**sep_conf["net_conf"]).to(device)
    classifier = DASResNetClassifier(**cls_conf["net_conf"]).to(device)
    load_model_state(separator, args.separator_ckpt, device, strict=False)
    load_model_state(classifier, args.classifier_ckpt, device, strict=False)
    for p in separator.parameters():
        p.requires_grad_(False)

    ds = dict(sep_conf["datasets"])
    ds["batch_size"] = min(int(ds.get("batch_size", 8)), 8)
    train_ds = dict(ds)
    train_ds["epoch_length"] = args.train_epoch_length
    train_ds["same_class_prob"] = args.train_same_class_prob
    if args.train_min_sources > 0:
        train_ds["min_sources"] = args.train_min_sources
    if args.train_max_sources > 0:
        train_ds["max_sources"] = args.train_max_sources
    if args.train_two_source_prob >= 0:
        train_ds["two_source_prob"] = args.train_two_source_prob
    if args.train_three_source_prob >= 0:
        train_ds["three_source_prob"] = args.train_three_source_prob
    val_ds = dict(ds)
    val_ds["epoch_length"] = args.val_epoch_length
    train_loader = make_das_mix_dataloader(split="train", **train_ds)
    val_loader = make_das_mix_dataloader(split="test", deterministic=True, **val_ds)

    ensure_dir(args.checkpoint_dir)
    logger = get_logger("sep_cls_finetune", str(Path(args.checkpoint_dir) / "train.log"))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1.0e-4)
    best_acc = 0.0
    logger.info(f"Device={device}, epochs={args.epochs}, lr={args.lr}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _, _ = run_epoch(separator, classifier, train_loader, optimizer, device, True, True)
        val_loss, val_acc, val_p, val_r, val_f1 = run_epoch(separator, classifier, val_loader, optimizer, device, False, False)
        logger.info(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )
        save_checkpoint(str(Path(args.checkpoint_dir) / "last.pt"), classifier, optimizer, None, epoch, {"best_acc": best_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(str(Path(args.checkpoint_dir) / "best.pt"), classifier, optimizer, None, epoch, {"best_acc": best_acc})


if __name__ == "__main__":
    main()
