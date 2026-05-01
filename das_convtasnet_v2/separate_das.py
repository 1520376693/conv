import argparse
import os
import yaml

import scipy.io as sio
import torch
from tqdm import tqdm

from DAS_Conv_TasNet import DASMCConvTasNet
from DAS_classifier import DASBaselineCNN
from das_preprocess import load_das_mat, crop_or_pad, moving_average_torch, to_cnn_input
from DASDataLoaders import ID_TO_CLASS
from utils import ensure_dir, load_model_state


def collect_mat_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    files = []
    for root, _, names in os.walk(input_path):
        for name in names:
            if name.lower().endswith(".mat"):
                files.append(os.path.join(root, name))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="options/train_das.yml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/separator/best.pt")
    parser.add_argument("--input", type=str, required=True, help="单个.mat文件或包含.mat的文件夹")
    parser.add_argument("--save_dir", type=str, default="results/separated_mat")
    parser.add_argument("--post_smooth_kernel", type=int, default=1)
    parser.add_argument("--cls_opt", type=str, default="")
    parser.add_argument("--cls_ckpt", type=str, default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    with open(args.opt, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DASMCConvTasNet(**opt["net_conf"]).to(device)
    load_model_state(model, args.checkpoint, device, strict=args.strict)
    model.eval()

    classifier = None
    if args.cls_opt and args.cls_ckpt:
        with open(args.cls_opt, "r", encoding="utf-8") as f:
            cls_opt = yaml.safe_load(f)
        classifier = DASBaselineCNN(**cls_opt["net_conf"]).to(device)
        load_model_state(classifier, args.cls_ckpt, device, strict=args.strict)
        classifier.eval()

    ensure_dir(args.save_dir)
    mat_files = collect_mat_files(args.input)
    if len(mat_files) == 0:
        raise RuntimeError(f"No .mat files found in {args.input}")

    ds_conf = opt["datasets"]
    chunk_size = ds_conf.get("chunk_size", 10000)
    amp_range = ds_conf.get("amp_range", 0.01)
    use_diff = ds_conf.get("use_diff", True)
    smooth_kernel = ds_conf.get("smooth_kernel", 1)

    with torch.no_grad():
        for path in tqdm(mat_files, ncols=120, desc="separate"):
            x = load_das_mat(
                path,
                use_diff=use_diff,
                norm_mode="amp",
                amp_range=amp_range,
                smooth_kernel=smooth_kernel,
            )
            x = crop_or_pad(x, chunk_size, random_crop=False)
            mix = x.unsqueeze(0).to(device)  # [1, 12, T]
            ests_t = model(mix)
            if args.post_smooth_kernel > 1:
                ests_t = moving_average_torch(ests_t, kernel_size=args.post_smooth_kernel)
            ests = ests_t.squeeze(0).cpu().numpy()  # [K, 12, T]

            base = os.path.splitext(os.path.basename(path))[0]
            sample_dir = os.path.join(args.save_dir, base)
            ensure_dir(sample_dir)
            sio.savemat(os.path.join(sample_dir, "mix.mat"), {"data": x.numpy().T})
            pred_lines = []
            for k in range(ests.shape[0]):
                out_name = f"est_{k+1}.mat"
                sio.savemat(os.path.join(sample_dir, out_name), {"data": ests[k].T})
                if classifier is not None:
                    est_tensor = torch.from_numpy(ests[k]).float().to(device)
                    logits = classifier(to_cnn_input(est_tensor).to(device))
                    prob = torch.softmax(logits, dim=1)[0]
                    pred = int(prob.argmax().item())
                    pred_lines.append(f"est_{k+1}: {ID_TO_CLASS[pred]} prob={prob[pred].item():.4f}")

            if pred_lines:
                with open(os.path.join(sample_dir, "predicted_labels.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(pred_lines) + "\n")

    print(f"Separated results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
