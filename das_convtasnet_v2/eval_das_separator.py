import argparse
import csv
import os
import yaml

import torch
from tqdm import tqdm

from DAS_Conv_TasNet import DASMCConvTasNet
from DASDataLoaders import make_das_mix_dataloader, ID_TO_CLASS
from DAS_loss import pit_si_snr_variable_sources, si_snr_pair_single, snr, sdr, mse, mae, pearson_corr
from utils import ensure_dir, load_model_state, to_device


def compute_metrics_with_best_info(mix, refs, ests, labels, n_src, best_infos):
    rows = []
    bsz = mix.shape[0]
    for b in range(bsz):
        n = int(n_src[b].item())
        info = best_infos[b]
        active_sum = torch.zeros_like(mix[b])
        for i, ref_idx in enumerate(info["perm"]):
            out_idx = info["out_ids"][i]
            ref = refs[b, ref_idx]
            est = ests[b, out_idx]
            mix_b = mix[b]
            active_sum = active_sum + est
            si_snr_est = si_snr_pair_single(est, ref).item()
            si_snr_mix = si_snr_pair_single(mix_b, ref).item()
            label_id = int(labels[b, ref_idx].item()) if labels is not None else -1
            rows.append({
                "sample": b,
                "source_index": i,
                "label_id": label_id,
                "label_name": ID_TO_CLASS.get(label_id, "unknown"),
                "n_src": n,
                "out_idx": out_idx,
                "ref_idx": ref_idx,
                "si_snr": si_snr_est,
                "si_snri": si_snr_est - si_snr_mix,
                "snr": snr(est, ref).item(),
                "sdr": sdr(est, ref).item(),
                "mse": mse(est, ref).item(),
                "mae": mae(est, ref).item(),
                "pcc": pearson_corr(est, ref).item(),
            })
        rows.append({
            "sample": b,
            "source_index": -1,
            "label_id": -1,
            "label_name": "active_sum",
            "n_src": n,
            "out_idx": -1,
            "ref_idx": -1,
            "si_snr": si_snr_pair_single(active_sum, mix[b]).item(),
            "si_snri": 0.0,
            "snr": snr(active_sum, mix[b]).item(),
            "sdr": sdr(active_sum, mix[b]).item(),
            "mse": mse(active_sum, mix[b]).item(),
            "mae": mae(active_sum, mix[b]).item(),
            "pcc": pearson_corr(active_sum, mix[b]).item(),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="options/train_das.yml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/separator/best.pt")
    parser.add_argument("--out_csv", type=str, default="results/metrics/separator_metrics.csv")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    with open(args.opt, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DASMCConvTasNet(**opt["net_conf"]).to(device)
    load_model_state(model, args.checkpoint, device, strict=args.strict)
    model.eval()

    loader = make_das_mix_dataloader(split="test", deterministic=True, **opt["datasets"])
    all_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, ncols=120, desc="eval_separator")):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch = to_device(batch, device)
            ests = model(batch["mix"])
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
            rows = compute_metrics_with_best_info(batch["mix"], batch["ref"], ests, batch.get("labels"), batch["n_src"], best_infos)
            for r in rows:
                r["batch"] = batch_idx
            all_rows.extend(rows)

    ensure_dir(os.path.dirname(args.out_csv))
    if len(all_rows) == 0:
        print("No rows generated.")
        return

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    active_rows = [r for r in all_rows if r["source_index"] >= 0]
    for key in ["si_snr", "si_snri", "snr", "sdr", "mse", "mae", "pcc"]:
        values = [r[key] for r in active_rows]
        print(f"{key}: mean={sum(values)/len(values):.4f}")

    print("\nPer-class mean SI-SNRi:")
    for name in sorted(set(r["label_name"] for r in active_rows)):
        vals = [r["si_snri"] for r in active_rows if r["label_name"] == name]
        print(f"  {name}: {sum(vals)/len(vals):.4f} ({len(vals)} sources)")

    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
