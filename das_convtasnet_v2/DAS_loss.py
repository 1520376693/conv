import itertools
import torch
import torch.nn.functional as F


def si_snr_pair_batch(est, ref, eps=1e-8):
    """
    est/ref: [B, C, T]
    return: [B]
    """
    b = est.shape[0]
    est = est.reshape(b, -1)
    ref = ref.reshape(b, -1)

    est = est - est.mean(dim=1, keepdim=True)
    ref = ref - ref.mean(dim=1, keepdim=True)

    dot = torch.sum(est * ref, dim=1)
    ref_energy = torch.sum(ref ** 2, dim=1) + eps
    est_energy = torch.sum(est ** 2, dim=1) + eps
    target_energy = dot ** 2 / ref_energy
    noise_energy = torch.clamp(est_energy - target_energy, min=eps)
    return 10.0 * torch.log10(target_energy / noise_energy + eps)


def si_snr_pair_single(est, ref, eps=1e-8):
    """est/ref: [C, T], return scalar."""
    return si_snr_pair_batch(est.unsqueeze(0), ref.unsqueeze(0), eps=eps)[0]


def pairwise_si_snr_global(ests, refs, eps=1e-8):
    """
    ests: [B, K, C, T], refs: [B, S, C, T]
    return: [B, K, S]
    """
    b, k, c, t = ests.shape
    s = refs.shape[1]

    e = ests.reshape(b, k, -1)
    r = refs.reshape(b, s, -1)
    e = e - e.mean(dim=2, keepdim=True)
    r = r - r.mean(dim=2, keepdim=True)

    dot = torch.einsum("bkd,bsd->bks", e, r)
    e_energy = torch.sum(e ** 2, dim=2).unsqueeze(2) + eps
    r_energy = torch.sum(r ** 2, dim=2).unsqueeze(1) + eps
    target_energy = dot ** 2 / r_energy
    noise_energy = torch.clamp(e_energy - target_energy, min=eps)
    return 10.0 * torch.log10(target_energy / noise_energy + eps)


def pairwise_si_snr_channelwise(ests, refs, eps=1e-8):
    """
    Channel-wise SI-SNR averaged over DAS spatial channels.
    This prevents one strong channel from dominating the PIT assignment.

    ests: [B, K, C, T], refs: [B, S, C, T]
    return: [B, K, S]
    """
    e = ests - ests.mean(dim=3, keepdim=True)
    r = refs - refs.mean(dim=3, keepdim=True)

    dot = torch.einsum("bkct,bsct->bksc", e, r)
    e_energy = torch.sum(e ** 2, dim=3).unsqueeze(2) + eps  # [B,K,1,C]
    r_energy = torch.sum(r ** 2, dim=3).unsqueeze(1) + eps  # [B,1,S,C]
    target_energy = dot ** 2 / r_energy
    noise_energy = torch.clamp(e_energy - target_energy, min=eps)
    sisnr = 10.0 * torch.log10(target_energy / noise_energy + eps)
    return sisnr.mean(dim=3)


def _relative_l1(est, ref, eps=1e-8):
    return (est - ref).abs().mean() / (ref.abs().mean() + eps)


def pit_si_snr_variable_sources(
    ests,
    refs,
    n_src,
    silence_weight=0.05,
    channel_weight=0.35,
    waveform_weight=0.10,
    mix=None,
    mix_consistency_weight=0.10,
    eps=1e-8,
):
    """
    可变源数、多通道PIT-SI-SNR。支持2/3/4源随机混合。

    ests: [B, K, C, T]
    refs: [B, K, C, T]，前n_src是真实源，后面补零
    n_src: [B]
    mix:  [B, C, T]，可选；用于active outputs的混合一致性约束

    返回:
        loss: scalar, 越小越好
        avg_sisnr: scalar, 越大越好
        best_infos: list，每个样本的最佳匹配关系，格式兼容原评估脚本
    """
    if ests.dim() != 4 or refs.dim() != 4:
        raise RuntimeError(f"ests/refs must be [B,K,C,T], got {ests.shape}/{refs.shape}")
    bsz, k, c, t = ests.shape
    if refs.shape[:2] != (bsz, k):
        raise RuntimeError(f"ests/refs source dims mismatch: {ests.shape}/{refs.shape}")

    global_score = pairwise_si_snr_global(ests, refs, eps=eps)
    if channel_weight > 0:
        channel_score = pairwise_si_snr_channelwise(ests, refs, eps=eps)
        pair_score = (1.0 - channel_weight) * global_score + channel_weight * channel_score
    else:
        pair_score = global_score

    total_loss = ests.new_tensor(0.0)
    total_score = ests.new_tensor(0.0)
    best_infos = []

    for b in range(bsz):
        n = int(n_src[b].item())
        if n < 1 or n > k:
            raise ValueError(f"n_src must be in [1,{k}], got {n}")

        output_combinations = list(itertools.combinations(range(k), n))
        permutations = list(itertools.permutations(range(n)))

        best_score = None
        best_out_ids = None
        best_perm = None

        for out_ids in output_combinations:
            for perm in permutations:
                score = ests.new_tensor(0.0)
                for i, ref_idx in enumerate(perm):
                    out_idx = out_ids[i]
                    score = score + pair_score[b, out_idx, ref_idx]
                score = score / n
                if best_score is None or score > best_score:
                    best_score = score
                    best_out_ids = out_ids
                    best_perm = perm

        loss_b = -best_score

        # Small waveform-domain term stabilizes early training when SI-SNR assignment is noisy.
        if waveform_weight > 0:
            est_active = torch.stack([ests[b, best_out_ids[i]] for i in range(n)], dim=0)
            ref_active = torch.stack([refs[b, best_perm[i]] for i in range(n)], dim=0)
            loss_b = loss_b + waveform_weight * _relative_l1(est_active, ref_active, eps=eps)

        # Encourage selected outputs to add back to the mixture. This is better than forcing all K
        # outputs to sum to mix because unused outputs are explicitly trained to be silent.
        if mix is not None and mix_consistency_weight > 0:
            active_sum = torch.stack([ests[b, idx] for idx in best_out_ids], dim=0).sum(dim=0)
            loss_b = loss_b + mix_consistency_weight * _relative_l1(active_sum, mix[b], eps=eps)

        unused_ids = [i for i in range(k) if i not in best_out_ids]
        if len(unused_ids) > 0 and silence_weight > 0:
            inactive = ests[b, unused_ids]
            denom = (mix[b].pow(2).mean() if mix is not None else refs[b, :n].pow(2).mean()) + eps
            silence_loss = inactive.pow(2).mean() / denom
            loss_b = loss_b + silence_weight * silence_loss

        total_loss = total_loss + loss_b
        total_score = total_score + best_score
        best_infos.append({"out_ids": best_out_ids, "perm": best_perm})

    return total_loss / bsz, total_score / bsz, best_infos


def snr(est, ref, eps=1e-8):
    err = est - ref
    return 10.0 * torch.log10(torch.sum(ref ** 2) / (torch.sum(err ** 2) + eps) + eps)


def sdr(est, ref, eps=1e-8):
    err = est - ref
    return 10.0 * torch.log10((ref.pow(2).sum() + eps) / (err.pow(2).sum() + eps))


def mse(est, ref):
    return torch.mean((est - ref) ** 2)


def mae(est, ref):
    return torch.mean(torch.abs(est - ref))


def pearson_corr(est, ref, eps=1e-8):
    x = est.reshape(-1) - est.reshape(-1).mean()
    y = ref.reshape(-1) - ref.reshape(-1).mean()
    return torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)) + eps)
