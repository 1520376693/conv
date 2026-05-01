import argparse
import os
import yaml

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from DAS_Conv_TasNet import DASMCConvTasNet
from DASDataLoaders import make_das_mix_dataloader
from DAS_loss import pit_si_snr_variable_sources
from utils import setup_seed, get_logger, ensure_dir, to_device, save_checkpoint, load_model_state, get_lr
from Conv_TasNet import check_parameters


class DASTrainer:
    def __init__(self, model, opt):
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() and len(opt.get("gpu_ids", [])) > 0 else "cpu")
        self.model = model.to(self.device)
        self.train_conf = opt["train"]
        self.checkpoint_dir = self.train_conf["checkpoint"]
        ensure_dir(self.checkpoint_dir)
        self.logger = get_logger("separator", os.path.join(self.checkpoint_dir, "train.log"))

        optimizer_name = self.train_conf.get("optimizer", "adamw").lower()
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), **opt["optimizer_kwargs"])
        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt["optimizer_kwargs"])
        else:
            raise ValueError("Only adam/adamw are supported in this script.")

        sched_name = self.train_conf.get("scheduler", "plateau").lower()
        if sched_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_conf.get("num_epochs", 120),
                eta_min=float(self.train_conf.get("min_lr", 1e-7)),
            )
        elif sched_name == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.train_conf.get("factor", 0.5),
                patience=self.train_conf.get("patience", 5),
                min_lr=float(self.train_conf.get("min_lr", 1e-8)),
            )
        else:
            raise ValueError("scheduler must be plateau or cosine")
        self.sched_name = sched_name

        self.clip_norm = self.train_conf.get("clip_norm", 5)
        self.num_epochs = self.train_conf.get("num_epochs", 120)
        self.use_amp = self.train_conf.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.start_epoch = 0
        self.best_val = float("inf")
        self.best_sisnr = -1e9

        resume_path = self.train_conf.get("resume", "")
        if resume_path:
            ckpt = load_model_state(self.model, resume_path, self.device, strict=self.train_conf.get("strict_resume", True))
            if "optim_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optim_state_dict"])
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.start_epoch = ckpt.get("epoch", 0)
            self.best_val = ckpt.get("best_val", self.best_val)
            self.best_sisnr = ckpt.get("best_sisnr", self.best_sisnr)
            self.logger.info(f"Resume from {resume_path}, epoch={self.start_epoch}")

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"AMP: {self.use_amp}")
        self.logger.info(f"Model parameters: {check_parameters(self.model):.2f} M")
        self.logger.info(f"Loss conf: {self.train_conf.get('loss', {})}")

    def _loss(self, ests, batch):
        loss_conf = self.train_conf.get("loss", {})
        return pit_si_snr_variable_sources(
            ests=ests,
            refs=batch["ref"],
            n_src=batch["n_src"],
            silence_weight=loss_conf.get("silence_weight", self.train_conf.get("silence_weight", 0.05)),
            channel_weight=loss_conf.get("channel_weight", 0.35),
            waveform_weight=loss_conf.get("waveform_weight", 0.10),
            mix=batch["mix"],
            mix_consistency_weight=loss_conf.get("mix_consistency_weight", 0.10),
        )

    def run_one_epoch(self, loader, training=True):
        self.model.train(training)
        losses, sisnrs = [], []

        pbar = tqdm(loader, ncols=120, desc="train_sep" if training else "valid_sep")
        for batch in pbar:
            batch = to_device(batch, self.device)
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    ests = self.model(batch["mix"])
                    loss, avg_sisnr, _ = self._loss(ests, batch)

                if training:
                    self.scaler.scale(loss).backward()
                    if self.clip_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            losses.append(loss.item())
            sisnrs.append(avg_sisnr.item())
            pbar.set_postfix(loss=sum(losses) / len(losses), sisnr=sum(sisnrs) / len(sisnrs), lr=f"{get_lr(self.optimizer):.2e}")

        return sum(losses) / len(losses), sum(sisnrs) / len(sisnrs)

    def fit(self, train_loader, val_loader):
        no_improve = 0
        early_stop = self.train_conf.get("early_stop", 25)

        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            train_loss, train_sisnr = self.run_one_epoch(train_loader, training=True)
            val_loss, val_sisnr = self.run_one_epoch(val_loader, training=False)

            if self.sched_name == "plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            self.logger.info(
                f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, train_si_snr={train_sisnr:.4f}, "
                f"val_loss={val_loss:.4f}, val_si_snr={val_sisnr:.4f}, lr={get_lr(self.optimizer):.3e}"
            )

            save_checkpoint(
                os.path.join(self.checkpoint_dir, "last.pt"),
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=epoch,
                extra={"best_val": self.best_val, "best_sisnr": self.best_sisnr},
            )

            improved = val_loss < self.best_val
            if improved:
                self.best_val = val_loss
                self.best_sisnr = max(self.best_sisnr, val_sisnr)
                no_improve = 0
                save_checkpoint(
                    os.path.join(self.checkpoint_dir, "best.pt"),
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=epoch,
                    extra={"best_val": self.best_val, "best_sisnr": self.best_sisnr},
                )
                self.logger.info(f"New best checkpoint saved. best_val={self.best_val:.4f}, best_sisnr={self.best_sisnr:.4f}")
            else:
                no_improve += 1
                self.logger.info(f"No improvement: {no_improve}/{early_stop}")
                if no_improve >= early_stop:
                    self.logger.info("Early stopping.")
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="options/train_das.yml")
    args = parser.parse_args()

    with open(args.opt, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    setup_seed(opt.get("seed", 2026), deterministic=opt.get("deterministic", False))
    gpu_ids = opt.get("gpu_ids", [0])
    if torch.cuda.is_available() and gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    model = DASMCConvTasNet(**opt["net_conf"])
    ds = opt["datasets"]

    train_loader = make_das_mix_dataloader(split="train", **ds)
    val_loader = make_das_mix_dataloader(split="test", **ds)

    trainer = DASTrainer(model, opt)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
