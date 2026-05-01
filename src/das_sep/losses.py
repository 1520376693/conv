from __future__ import annotations

import itertools

import torch


def si_snr_pair_batch(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b = est.shape[0]
    est = est.reshape(b, -1)
    ref = ref.reshape(b, -1)
    est = est - est.mean(dim=1, keepdim=True)
    ref = ref - ref.mean(dim=1, keepdim=True)
    dot = torch.sum(est * ref, dim=1)
    ref_energy = torch.sum(ref**2, dim=1) + eps
    est_energy = torch.sum(est**2, dim=1) + eps
    target_energy = dot**2 / ref_energy
    noise_energy = torch.clamp(est_energy - target_energy, min=eps)
    return 10.0 * torch.log10(target_energy / noise_energy + eps)


def si_snr_pair_single(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return si_snr_pair_batch(est.unsqueeze(0), ref.unsqueeze(0), eps=eps)[0]


def pairwise_si_snr_global(ests: torch.Tensor, refs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b, k, _, _ = ests.shape
    s = refs.shape[1]
    e = ests.reshape(b, k, -1).float()
    r = refs.reshape(b, s, -1).float()
    e = e - e.mean(dim=2, keepdim=True)
    r = r - r.mean(dim=2, keepdim=True)
    dot = torch.einsum("bkd,bsd->bks", e, r)
    e_energy = e.pow(2).sum(dim=2).unsqueeze(2) + eps
    r_energy = r.pow(2).sum(dim=2).unsqueeze(1) + eps
    target = torch.clamp(dot.pow(2) / r_energy, min=eps)
    noise = torch.clamp(e_energy - target, min=eps)
    return 10.0 * torch.log10(target / noise + eps)


def pairwise_si_snr_channelwise(ests: torch.Tensor, refs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ests = ests.float()
    refs = refs.float()
    e = ests - ests.mean(dim=3, keepdim=True)
    r = refs - refs.mean(dim=3, keepdim=True)
    dot = torch.einsum("bkct,bsct->bksc", e, r)
    e_energy = e.pow(2).sum(dim=3).unsqueeze(2) + eps
    r_energy = r.pow(2).sum(dim=3).unsqueeze(1) + eps
    target = torch.clamp(dot.pow(2) / r_energy, min=eps)
    noise = torch.clamp(e_energy - target, min=eps)
    return (10.0 * torch.log10(target / noise + eps)).mean(dim=3)


def relative_l1(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (est - ref).abs().mean() / (ref.abs().mean() + eps)


def stft_mag_loss(est: torch.Tensor, ref: torch.Tensor, fft_sizes=(256, 512, 1024), eps: float = 1e-8) -> torch.Tensor:
    """Multi-resolution magnitude loss for [S,C,T] or [B,S,C,T] tensors."""
    if est.dim() == 3:
        est = est.unsqueeze(0)
        ref = ref.unsqueeze(0)
    b, s, c, t = est.shape
    est = est.reshape(b * s * c, t).float()
    ref = ref.reshape(b * s * c, t).float()
    total = est.new_tensor(0.0)
    used = 0
    for n_fft in fft_sizes:
        if t < n_fft:
            continue
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=est.device, dtype=est.dtype)
        est_mag = torch.stft(est, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True).abs()
        ref_mag = torch.stft(ref, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True).abs()
        total = total + torch.mean(torch.abs(torch.log(est_mag + eps) - torch.log(ref_mag + eps)))
        used += 1
    return total / max(used, 1)


def pit_si_snr_variable_sources(
    ests: torch.Tensor,
    refs: torch.Tensor,
    n_src: torch.Tensor,
    silence_weight: float = 0.05,
    channel_weight: float = 0.35,
    waveform_weight: float = 0.10,
    stft_weight: float = 0.0,
    mix: torch.Tensor | None = None,
    mix_consistency_weight: float = 0.10,
    eps: float = 1e-8,
):
    if ests.dim() != 4 or refs.dim() != 4:
        raise RuntimeError(f"ests/refs must be [B,K,C,T], got {ests.shape}/{refs.shape}")
    bsz, k, _, _ = ests.shape
    if refs.shape[0] != bsz:
        raise RuntimeError(f"Source dims mismatch: {ests.shape}/{refs.shape}")
    if int(n_src.max().item()) > refs.shape[1]:
        raise RuntimeError(f"n_src exceeds reference sources: n_src={n_src.tolist()}, refs={refs.shape}")

    global_score = pairwise_si_snr_global(ests, refs, eps=eps)
    if channel_weight > 0:
        pair_score = (1.0 - channel_weight) * global_score + channel_weight * pairwise_si_snr_channelwise(ests, refs, eps=eps)
    else:
        pair_score = global_score

    total_loss = ests.new_tensor(0.0)
    total_score = ests.new_tensor(0.0)
    best_infos = []
    for b in range(bsz):
        n = int(n_src[b].item())
        output_combinations = list(itertools.combinations(range(k), n))
        permutations = list(itertools.permutations(range(n)))
        best_score, best_out_ids, best_perm = None, None, None
        for out_ids in output_combinations:
            for perm in permutations:
                score = ests.new_tensor(0.0)
                for i, ref_idx in enumerate(perm):
                    score = score + pair_score[b, out_ids[i], ref_idx]
                score = score / n
                if best_score is None or score > best_score:
                    best_score, best_out_ids, best_perm = score, out_ids, perm
        loss_b = -best_score
        est_active = torch.stack([ests[b, best_out_ids[i]] for i in range(n)], dim=0)
        ref_active = torch.stack([refs[b, best_perm[i]] for i in range(n)], dim=0)
        if waveform_weight > 0:
            loss_b = loss_b + waveform_weight * relative_l1(est_active, ref_active, eps=eps)
        if stft_weight > 0:
            loss_b = loss_b + stft_weight * stft_mag_loss(est_active, ref_active, eps=eps)
        if mix is not None and mix_consistency_weight > 0:
            loss_b = loss_b + mix_consistency_weight * relative_l1(est_active.sum(dim=0), mix[b], eps=eps)
        unused_ids = [i for i in range(k) if i not in best_out_ids]
        if unused_ids and silence_weight > 0:
            denom = (mix[b].pow(2).mean() if mix is not None else refs[b, :n].pow(2).mean()) + eps
            loss_b = loss_b + silence_weight * ests[b, unused_ids].pow(2).mean() / denom
        total_loss = total_loss + loss_b
        total_score = total_score + best_score
        best_infos.append({"out_ids": best_out_ids, "perm": best_perm})
    return total_loss / bsz, total_score / bsz, best_infos


def snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 10.0 * torch.log10(ref.pow(2).sum() / ((est - ref).pow(2).sum() + eps) + eps)


def sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return snr(est, ref, eps=eps)


def mse(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return torch.mean((est - ref) ** 2)


def mae(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(est - ref))


def pearson_corr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = est.reshape(-1) - est.reshape(-1).mean()
    y = ref.reshape(-1) - ref.reshape(-1).mean()
    return torch.sum(x * y) / (torch.sqrt(torch.sum(x**2)) * torch.sqrt(torch.sum(y**2)) + eps)
