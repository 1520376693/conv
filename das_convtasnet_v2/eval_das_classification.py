import argparse
import os
import yaml

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from DAS_Conv_TasNet import DASMCConvTasNet
from DAS_classifier import DASBaselineCNN
from DASDataLoaders import make_das_mix_dataloader, ID_TO_CLASS
from DAS_loss import pit_si_snr_variable_sources
from das_preprocess import to_cnn_input, moving_average_torch
from utils import load_model_state, to_device, ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sep_opt", type=str, default="options/train_das.yml")
    parser.add_argument("--cls_opt", type=str, default="options/train_classifier.yml")
    parser.add_argument("--sep_ckpt", type=str, default="checkpoints/separator/best.pt")
    parser.add_argument("--cls_ckpt", type=str, default="checkpoints/classifier/best.pt")
    parser.add_argument("--out_dir", type=str, default="results/metrics")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--post_smooth_kernel", type=int, default=1, help=">1时对分离输出做时间维移动平均，降低高频伪影")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    with open(args.sep_opt, "r", encoding="utf-8") as f:
        sep_opt = yaml.safe_load(f)
    with open(args.cls_opt, "r", encoding="utf-8") as f:
        cls_opt = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    separator = DASMCConvTasNet(**sep_opt["net_conf"]).to(device)
    classifier = DASBaselineCNN(**cls_opt["net_conf"]).to(device)
    load_model_state(separator, args.sep_ckpt, device, strict=args.strict)
    load_model_state(classifier, args.cls_ckpt, device, strict=args.strict)
    separator.eval()
    classifier.eval()

    loader = make_das_mix_dataloader(split="test", deterministic=True, **sep_opt["datasets"])
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, ncols=120, desc="eval_classification")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch = to_device(batch, device)
            ests = separator(batch["mix"])
            if args.post_smooth_kernel > 1:
                ests = moving_average_torch(ests, kernel_size=args.post_smooth_kernel)
            _, _, best_infos = pit_si_snr_variable_sources(
                ests,
                batch["ref"],
                batch["n_src"],
                silence_weight=0.0,
                channel_weight=0.35,
                waveform_weight=0.0,
                mix=batch["mix"],
                mix_consistency_weight=0.0,
            )

            for b, info in enumerate(best_infos):
                for i, ref_idx in enumerate(info["perm"]):
                    out_idx = info["out_ids"][i]
                    est = ests[b, out_idx]  # [12, T]
                    x = to_cnn_input(est).to(device)  # [1, 1, T, 12]
                    logits = classifier(x)
                    pred = int(logits.argmax(dim=1).item())
                    true = int(batch["labels"][b, ref_idx].item())
                    if true < 6:
                        y_true.append(true)
                        y_pred.append(pred)

    ensure_dir(args.out_dir)
    if len(y_true) == 0:
        print("No valid labels were collected.")
        return

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))

    np.savetxt(os.path.join(args.out_dir, "separated_classification_confusion.csv"), cm, fmt="%d", delimiter=",")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(6)),
        target_names=[ID_TO_CLASS[i] for i in range(6)],
        zero_division=0,
    )
    with open(os.path.join(args.out_dir, "separated_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\naccuracy={acc:.6f}\nprecision={p:.6f}\nrecall={r:.6f}\nf1={f1:.6f}\n")

    print(report)
    print(f"Accuracy={acc:.4f}, Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
