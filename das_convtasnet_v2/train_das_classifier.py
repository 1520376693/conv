import argparse
import os
import yaml

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

from DAS_classifier import DASBaselineCNN
from DASDataLoaders import make_single_event_dataloader
from utils import setup_seed, get_logger, ensure_dir, to_device, save_checkpoint, load_model_state, get_lr


class ClassifierTrainer:
    def __init__(self, model, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() and len(opt.get("gpu_ids", [])) > 0 else "cpu")
        self.model = model.to(self.device)
        self.conf = opt["train"]
        self.checkpoint_dir = self.conf["checkpoint"]
        ensure_dir(self.checkpoint_dir)
        self.logger = get_logger("classifier", os.path.join(self.checkpoint_dir, "train.log"))

        optimizer_name = self.conf.get("optimizer", "adamw").lower()
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), **opt["optimizer_kwargs"])
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt["optimizer_kwargs"])
        else:
            raise ValueError("Only adam/adamw are supported.")

        sched_name = self.conf.get("scheduler", "plateau").lower()
        if sched_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.conf.get("num_epochs", 120),
                eta_min=float(self.conf.get("min_lr", 1e-7)),
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.conf.get("factor", 0.5),
                patience=self.conf.get("patience", 5),
                min_lr=float(self.conf.get("min_lr", 1e-8)),
            )
        self.sched_name = sched_name
        self.num_epochs = self.conf.get("num_epochs", 120)
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.start_epoch = 0
        self.label_smoothing = self.conf.get("label_smoothing", 0.05)

        resume_path = self.conf.get("resume", "")
        if resume_path:
            ckpt = load_model_state(self.model, resume_path, self.device, strict=self.conf.get("strict_resume", True))
            if "optim_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optim_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0)
            self.best_acc = ckpt.get("best_acc", 0.0)
            self.best_f1 = ckpt.get("best_f1", 0.0)

    def run_one_epoch(self, loader, training=True):
        self.model.train(training)
        losses, preds, labels = [], [], []
        pbar = tqdm(loader, ncols=120, desc="train_cls" if training else "valid_cls")

        for batch in pbar:
            batch = to_device(batch, self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = self.model(batch["x"])
                loss = F.cross_entropy(logits, batch["label"], label_smoothing=self.label_smoothing)
                if training:
                    loss.backward()
                    clip_norm = self.conf.get("clip_norm", 5)
                    if clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    self.optimizer.step()

            pred = logits.argmax(dim=1)
            losses.append(loss.item())
            preds.extend(pred.detach().cpu().tolist())
            labels.extend(batch["label"].detach().cpu().tolist())
            pbar.set_postfix(loss=sum(losses) / len(losses), acc=accuracy_score(labels, preds), lr=f"{get_lr(self.optimizer):.2e}")

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        cm = confusion_matrix(labels, preds, labels=list(range(6)))
        return sum(losses) / len(losses), acc, precision, recall, f1, cm

    def fit(self, train_loader, val_loader):
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            train_loss, train_acc, train_p, train_r, train_f1, _ = self.run_one_epoch(train_loader, True)
            val_loss, val_acc, val_p, val_r, val_f1, cm = self.run_one_epoch(val_loader, False)
            if self.sched_name == "plateau":
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            self.logger.info(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, lr={get_lr(self.optimizer):.3e}"
            )
            np.savetxt(os.path.join(self.checkpoint_dir, "confusion_matrix_last.csv"), cm, fmt="%d", delimiter=",")

            save_checkpoint(
                os.path.join(self.checkpoint_dir, "last.pt"),
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=epoch,
                extra={"best_acc": self.best_acc, "best_f1": self.best_f1},
            )
            # Primary: accuracy. F1 is used to break ties and protect minority classes.
            if val_acc > self.best_acc or (abs(val_acc - self.best_acc) < 1e-6 and val_f1 > self.best_f1):
                self.best_acc = val_acc
                self.best_f1 = val_f1
                save_checkpoint(
                    os.path.join(self.checkpoint_dir, "best.pt"),
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=epoch,
                    extra={"best_acc": self.best_acc, "best_f1": self.best_f1},
                )
                np.savetxt(os.path.join(self.checkpoint_dir, "confusion_matrix_best.csv"), cm, fmt="%d", delimiter=",")
                self.logger.info(f"New best classifier saved. best_acc={self.best_acc:.4f}, best_f1={self.best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="options/train_classifier.yml")
    args = parser.parse_args()

    with open(args.opt, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    setup_seed(opt.get("seed", 2026), deterministic=opt.get("deterministic", False))
    gpu_ids = opt.get("gpu_ids", [0])
    if torch.cuda.is_available() and gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    model = DASBaselineCNN(**opt["net_conf"])
    ds = opt["datasets"]
    train_loader = make_single_event_dataloader(split="train", **ds)
    val_loader = make_single_event_dataloader(split="test", **ds)

    trainer = ClassifierTrainer(model, opt)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
