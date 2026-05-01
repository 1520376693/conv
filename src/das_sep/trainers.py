from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from .losses import pit_si_snr_variable_sources
from .utils import count_parameters, ensure_dir, get_logger, get_lr, load_model_state, save_checkpoint, to_device


def build_optimizer(model, conf: dict):
    name = conf.get("optimizer", "adamw").lower()
    kwargs = conf.get("optimizer_kwargs", {"lr": 8e-4, "weight_decay": 1e-4})
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError("optimizer must be adam or adamw")


def build_scheduler(optimizer, conf: dict, mode: str):
    name = conf.get("scheduler", "plateau").lower()
    if name == "cosine":
        return name, CosineAnnealingLR(optimizer, T_max=conf.get("num_epochs", 120), eta_min=float(conf.get("min_lr", 1e-7)))
    return name, ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=conf.get("factor", 0.5),
        patience=conf.get("patience", 6),
        min_lr=float(conf.get("min_lr", 1e-8)),
    )


class SeparatorTrainer:
    def __init__(self, model, opt: dict, device: torch.device):
        self.opt = opt
        self.device = device
        self.model = model.to(device)
        self.conf = opt["train"]
        self.checkpoint_dir = self.conf["checkpoint"]
        ensure_dir(self.checkpoint_dir)
        self.logger = get_logger("separator", os.path.join(self.checkpoint_dir, "train.log"))
        self.optimizer = build_optimizer(self.model, self.conf)
        self.sched_name, self.scheduler = build_scheduler(self.optimizer, self.conf, mode="min")
        self.use_amp = self.conf.get("use_amp", True) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.clip_norm = self.conf.get("clip_norm", 5)
        self.num_epochs = self.conf.get("num_epochs", 120)
        self.best_val = float("inf")
        self.best_sisnr = -1e9
        self.start_epoch = 0
        self.skipped_nonfinite = 0
        resume = self.conf.get("resume", "")
        if resume:
            ckpt = load_model_state(self.model, resume, device, strict=self.conf.get("strict_resume", True))
            if "optim_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optim_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0)
            self.best_val = ckpt.get("best_val", self.best_val)
            self.best_sisnr = ckpt.get("best_sisnr", self.best_sisnr)
        self.logger.info(f"Device={device}, AMP={self.use_amp}, params={count_parameters(model):.2f}M")

    def _loss(self, ests, batch):
        return pit_si_snr_variable_sources(ests=ests, refs=batch["ref"], n_src=batch["n_src"], mix=batch["mix"], **self.conf.get("loss", {}))

    def run_one_epoch(self, loader, training=True):
        self.model.train(training)
        losses, sisnrs = [], []
        desc = "train_sep" if training else "valid_sep"
        for batch in tqdm(loader, ncols=120, desc=desc):
            batch = to_device(batch, self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    ests = self.model(batch["mix"])
                    loss, avg_sisnr, _ = self._loss(ests, batch)
                if not torch.isfinite(loss):
                    self.skipped_nonfinite += 1
                    self.logger.warning(f"Skip non-finite separator loss. skipped={self.skipped_nonfinite}")
                    if training:
                        self.optimizer.zero_grad(set_to_none=True)
                    continue
                if training:
                    self.scaler.scale(loss).backward()
                    if self.clip_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            if torch.isfinite(avg_sisnr):
                losses.append(float(loss.detach().cpu()))
                sisnrs.append(float(avg_sisnr.detach().cpu()))
        if not losses:
            return float("inf"), float("-inf")
        return float(np.mean(losses)), float(np.mean(sisnrs))

    def fit(self, train_loader, val_loader):
        no_improve = 0
        early_stop = self.conf.get("early_stop", 25)
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            train_loss, train_sisnr = self.run_one_epoch(train_loader, True)
            val_loss, val_sisnr = self.run_one_epoch(val_loader, False)
            self.scheduler.step(val_loss) if self.sched_name == "plateau" else self.scheduler.step()
            self.logger.info(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_si_snr={train_sisnr:.4f}, "
                f"val_loss={val_loss:.4f}, val_si_snr={val_sisnr:.4f}, lr={get_lr(self.optimizer):.3e}"
            )
            save_checkpoint(os.path.join(self.checkpoint_dir, "last.pt"), self.model, self.optimizer, self.scheduler, epoch, {"best_val": self.best_val, "best_sisnr": self.best_sisnr})
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.best_sisnr = max(self.best_sisnr, val_sisnr)
                no_improve = 0
                save_checkpoint(os.path.join(self.checkpoint_dir, "best.pt"), self.model, self.optimizer, self.scheduler, epoch, {"best_val": self.best_val, "best_sisnr": self.best_sisnr})
            else:
                no_improve += 1
                if no_improve >= early_stop:
                    self.logger.info("Early stopping.")
                    break


class ClassifierTrainer:
    def __init__(self, model, opt: dict, device: torch.device):
        self.opt = opt
        self.device = device
        self.model = model.to(device)
        self.conf = opt["train"]
        self.checkpoint_dir = self.conf["checkpoint"]
        ensure_dir(self.checkpoint_dir)
        self.logger = get_logger("classifier", os.path.join(self.checkpoint_dir, "train.log"))
        self.optimizer = build_optimizer(self.model, self.conf)
        self.sched_name, self.scheduler = build_scheduler(self.optimizer, self.conf, mode="max")
        self.num_epochs = self.conf.get("num_epochs", 120)
        self.label_smoothing = self.conf.get("label_smoothing", 0.05)
        self.clip_norm = self.conf.get("clip_norm", 5)
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.start_epoch = 0
        self.logger.info(f"Device={device}, params={count_parameters(model):.2f}M")

    def run_one_epoch(self, loader, training=True):
        self.model.train(training)
        losses, preds, labels = [], [], []
        desc = "train_cls" if training else "valid_cls"
        for batch in tqdm(loader, ncols=120, desc=desc):
            batch = to_device(batch, self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits = self.model(batch["x"])
                loss = F.cross_entropy(logits, batch["label"], label_smoothing=self.label_smoothing)
                if training:
                    loss.backward()
                    if self.clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            labels.extend(batch["label"].detach().cpu().tolist())
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        cm = confusion_matrix(labels, preds, labels=list(range(6)))
        return float(np.mean(losses)), acc, precision, recall, f1, cm

    def fit(self, train_loader, val_loader):
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            train_loss, train_acc, _, _, _, _ = self.run_one_epoch(train_loader, True)
            val_loss, val_acc, val_p, val_r, val_f1, cm = self.run_one_epoch(val_loader, False)
            self.scheduler.step(val_acc) if self.sched_name == "plateau" else self.scheduler.step()
            self.logger.info(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, lr={get_lr(self.optimizer):.3e}"
            )
            np.savetxt(os.path.join(self.checkpoint_dir, "confusion_matrix_last.csv"), cm, fmt="%d", delimiter=",")
            save_checkpoint(os.path.join(self.checkpoint_dir, "last.pt"), self.model, self.optimizer, self.scheduler, epoch, {"best_acc": self.best_acc, "best_f1": self.best_f1})
            if val_acc > self.best_acc or (abs(val_acc - self.best_acc) < 1e-6 and val_f1 > self.best_f1):
                self.best_acc = val_acc
                self.best_f1 = val_f1
                save_checkpoint(os.path.join(self.checkpoint_dir, "best.pt"), self.model, self.optimizer, self.scheduler, epoch, {"best_acc": self.best_acc, "best_f1": self.best_f1})
                np.savetxt(os.path.join(self.checkpoint_dir, "confusion_matrix_best.csv"), cm, fmt="%d", delimiter=",")
