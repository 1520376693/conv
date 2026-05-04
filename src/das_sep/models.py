from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(1, 2), keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int):
        super().__init__(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def select_norm(norm: str, dim: int) -> nn.Module:
    norm = norm.lower()
    if norm == "gln":
        return GlobalLayerNorm(dim)
    if norm == "cln":
        return CumulativeLayerNorm(dim)
    if norm == "bn":
        return nn.BatchNorm1d(dim)
    raise ValueError("norm must be gln, cln, or bn")


class TemporalBlockWithSkip(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, dilation: int, norm="gln", dropout=0.0):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1, bias=False),
            nn.PReLU(),
            select_norm(norm, hidden_channels),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                padding=pad,
                dilation=dilation,
                groups=hidden_channels,
                bias=False,
            ),
            nn.PReLU(),
            select_norm(norm, hidden_channels),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.res_out = nn.Conv1d(hidden_channels, channels, 1)
        self.skip_out = nn.Conv1d(hidden_channels, channels, 1)

    def forward(self, x: torch.Tensor):
        y = self.net(x)
        return x + self.res_out(y), self.skip_out(y)


class ChannelSE1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DASSpatialStem(nn.Module):
    """Small spatial-temporal front-end for adjacent DAS channels."""

    def __init__(self, channels: int = 12, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
                    nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(channels),
                    nn.SiLU(inplace=True),
                    ChannelSE1D(channels),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DASSpatialStem2D(nn.Module):
    """2D time-channel stem for local DAS propagation patterns."""

    def __init__(self, channels: int = 12, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = max(channels * 2, 16)
        blocks = [
            nn.Conv2d(1, hidden, kernel_size=(9, 3), padding=(4, 1), bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ]
        for _ in range(depth):
            blocks.extend(
                [
                    nn.Conv2d(hidden, hidden, kernel_size=(7, 3), padding=(3, 1), groups=hidden, bias=False),
                    nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.SiLU(inplace=True),
                    nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
        blocks.append(nn.Conv2d(hidden, 1, kernel_size=1, bias=False))
        self.net = nn.Sequential(*blocks)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2).unsqueeze(1)
        y = self.net(y).squeeze(1).transpose(1, 2)
        return x + y


class DASMCConvTasNet(nn.Module):
    def __init__(
        self,
        in_channels=12,
        out_channels=12,
        N=256,
        L=20,
        B=256,
        H=512,
        P=3,
        X=8,
        R=4,
        norm="gln",
        max_sources=4,
        activate="relu",
        dropout=0.05,
        use_encoder_relu=True,
        spatial_stem=True,
        spatial_stem_type="1d",
        spatial_depth=2,
        mixture_consistency=False,
        encoder_scales=None,
    ):
        super().__init__()
        if L % 2 != 0:
            raise ValueError("L must be even because stride is L//2")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_sources = max_sources
        self.N = N
        self.activate = activate.lower()
        self.use_encoder_relu = use_encoder_relu
        self.mixture_consistency = mixture_consistency
        scales = [int(scale) for scale in (encoder_scales or [])]
        if scales:
            if any(scale % 2 != 0 or scale <= 0 for scale in scales):
                raise ValueError("encoder_scales must contain positive even kernel sizes")
            scales = [L] + [scale for scale in scales if scale != L]
        else:
            scales = [L]
        self.encoder_scales = scales
        self.multiscale_encoder = len(scales) > 1
        self.encoder_dim = N * len(scales)

        stem_type = str(spatial_stem_type).lower()
        if not spatial_stem:
            self.spatial_stem = nn.Identity()
        elif stem_type in {"1d", "conv1d"}:
            self.spatial_stem = DASSpatialStem(in_channels, depth=spatial_depth, dropout=dropout)
        elif stem_type in {"2d", "conv2d"}:
            self.spatial_stem = DASSpatialStem2D(in_channels, depth=spatial_depth, dropout=dropout)
        else:
            raise ValueError("spatial_stem_type must be 1d or 2d")
        self.encoder = nn.Conv1d(in_channels, N, kernel_size=L, stride=L // 2, bias=False)
        extra_encoders = []
        extra_decoders = []
        for scale in scales[1:]:
            padding = max(0, (scale - L) // 2)
            extra_encoders.append(
                nn.Conv1d(in_channels, N, kernel_size=scale, stride=L // 2, padding=padding, bias=False)
            )
            extra_decoders.append(
                nn.ConvTranspose1d(N, out_channels, kernel_size=scale, stride=L // 2, padding=padding, bias=False)
            )
        self.extra_encoders = nn.ModuleList(extra_encoders)
        self.extra_decoders = nn.ModuleList(extra_decoders)
        self.layer_norm = select_norm("cln", self.encoder_dim)
        self.bottleneck = nn.Conv1d(self.encoder_dim, B, 1, bias=False)
        self.separation = nn.ModuleList(
            [
                TemporalBlockWithSkip(B, H, P, dilation=2**i, norm=norm, dropout=dropout)
                for _ in range(R)
                for i in range(X)
            ]
        )
        self.mask_prelu = nn.PReLU()
        self.mask_norm = select_norm(norm, B)
        self.gen_masks = nn.Conv1d(B, max_sources * self.encoder_dim, 1)
        self.decoder = nn.ConvTranspose1d(N, out_channels, kernel_size=L, stride=L // 2, bias=False)

        if self.activate not in {"relu", "sigmoid", "softmax"}:
            raise ValueError("activate must be relu, sigmoid, or softmax")

    @staticmethod
    def _align_frames(x: torch.Tensor, target_frames: int) -> torch.Tensor:
        if x.shape[-1] == target_frames:
            return x
        return F.interpolate(x, size=target_frames, mode="linear", align_corners=False)

    @staticmethod
    def _crop_or_pad_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        if x.shape[-1] > target_t:
            return x[..., :target_t]
        if x.shape[-1] < target_t:
            return F.pad(x, (0, target_t - x.shape[-1]))
        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        base = self.encoder(x)
        target_frames = base.shape[-1]
        features = [base]
        for encoder in self.extra_encoders:
            features.append(self._align_frames(encoder(x), target_frames))
        w = torch.cat(features, dim=1) if len(features) > 1 else base
        return F.relu(w) if self.use_encoder_relu else w

    def _decode_source(self, masked_features: torch.Tensor, target_t: int) -> torch.Tensor:
        if not self.multiscale_encoder:
            return self._crop_or_pad_time(self.decoder(masked_features), target_t)
        chunks = masked_features.split(self.N, dim=1)
        estimates = [self._crop_or_pad_time(self.decoder(chunks[0]), target_t)]
        for chunk, decoder in zip(chunks[1:], self.extra_decoders):
            estimates.append(self._crop_or_pad_time(decoder(chunk), target_t))
        return torch.stack(estimates, dim=0).mean(dim=0)

    def _activate_masks(self, masks: torch.Tensor) -> torch.Tensor:
        if self.activate == "relu":
            return F.relu(masks)
        if self.activate == "sigmoid":
            return torch.sigmoid(masks)
        return torch.softmax(masks, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise RuntimeError(f"Expected [B,C,T], got {x.shape}")
        bsz, channels, t = x.shape
        if channels != self.in_channels:
            raise RuntimeError(f"Expected {self.in_channels} channels, got {channels}")
        mix = x
        x = self.spatial_stem(x)
        w = self._encode(x)
        y = self.bottleneck(self.layer_norm(w))
        skip_sum = None
        for block in self.separation:
            y, skip = block(y)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        m = self.mask_norm(self.mask_prelu(skip_sum))
        masks = self.gen_masks(m).view(bsz, self.max_sources, self.encoder_dim, -1)
        masks = self._activate_masks(masks)
        ests = []
        for k in range(self.max_sources):
            ests.append(self._decode_source(w * masks[:, k], t))
        ests = torch.stack(ests, dim=1)
        if self.mixture_consistency:
            ests = ests + (mix.unsqueeze(1) - ests.sum(dim=1, keepdim=True)) / float(self.max_sources)
        return ests


class SEBlock2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class Residual2DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1), dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(9, 3), stride=stride, padding=(4, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(7, 3), padding=(3, 1), bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == (1, 1)
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.drop(self.conv(x)) + self.shortcut(x))


class DASResNetClassifier(nn.Module):
    def __init__(self, num_classes=6, base_channels=40, dropout=0.35):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=(101, 3), stride=(8, 1), padding=(50, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
        )
        self.stage1 = nn.Sequential(Residual2DBlock(c, c, dropout=0.05), Residual2DBlock(c, c, dropout=0.05))
        self.stage2 = nn.Sequential(Residual2DBlock(c, c * 2, stride=(2, 1), dropout=0.08), Residual2DBlock(c * 2, c * 2, dropout=0.08))
        self.stage3 = nn.Sequential(Residual2DBlock(c * 2, c * 4, stride=(2, 1), dropout=0.12), Residual2DBlock(c * 4, c * 4, dropout=0.12), SEBlock2D(c * 4))
        self.stage4 = nn.Sequential(Residual2DBlock(c * 4, c * 4, dropout=0.15), SEBlock2D(c * 4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c * 4, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)
