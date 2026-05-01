import torch
import torch.nn as nn
import torch.nn.functional as F

from Conv_TasNet import Conv1D, ConvTrans1D, TemporalBlockWithSkip, select_norm


class DASMCConvTasNet(nn.Module):
    """
    Multi-channel Conv-TasNet for 12-channel DAS spatial-temporal vibration signals.

    Input:
        x: [B, 12, T]
    Output:
        ests: [B, K, 12, T]

    Key upgrades over the first version:
        1) proper TCN skip connections for mask generation;
        2) optional spatial channel mixing before encoder;
        3) safer mask activations: relu/sigmoid/softmax;
        4) source-count independent fixed K outputs, suitable for 2/3/4-source PIT.
    """

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
        max_sources=3,
        activate="relu",
        causal=False,
        dropout=0.0,
        use_encoder_relu=True,
        use_spatial_stem=True,
        mixture_consistency=False,
    ):
        super().__init__()
        if L % 2 != 0:
            raise ValueError("L should be even because stride is L//2.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_sources = max_sources
        self.N = N
        self.L = L
        self.activate = activate.lower()
        self.use_encoder_relu = use_encoder_relu
        self.mixture_consistency = mixture_consistency

        # A light spatial-temporal stem helps the network exploit the 12 adjacent fiber channels
        # before chunk-level Conv-TasNet encoding.
        self.spatial_stem = (
            nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True),
                nn.PReLU(),
            )
            if use_spatial_stem
            else nn.Identity()
        )

        self.encoder = Conv1D(in_channels, N, kernel_size=L, stride=L // 2, padding=0, bias=False)
        self.layer_norm = select_norm("cln", N)
        self.bottleneck = Conv1D(N, B, 1, bias=False)

        blocks = []
        for _ in range(R):
            for i in range(X):
                blocks.append(
                    TemporalBlockWithSkip(
                        channels=B,
                        hidden_channels=H,
                        kernel_size=P,
                        dilation=2 ** i,
                        norm=norm,
                        causal=causal,
                        dropout=dropout,
                    )
                )
        self.separation = nn.ModuleList(blocks)

        self.mask_prelu = nn.PReLU()
        self.mask_norm = select_norm(norm, B)
        self.gen_masks = Conv1D(B, max_sources * N, 1)
        self.decoder = ConvTrans1D(N, out_channels, kernel_size=L, stride=L // 2, bias=False)

        if self.activate not in {"relu", "sigmoid", "softmax"}:
            raise ValueError("activate must be relu, sigmoid, or softmax")

    def _apply_mask_activation(self, masks):
        # masks: [B, K, N, T']
        if self.activate == "relu":
            return F.relu(masks)
        if self.activate == "sigmoid":
            return torch.sigmoid(masks)
        # softmax is useful when all K outputs should share all mixture energy;
        # for variable-source training, relu/sigmoid is usually better.
        return torch.softmax(masks, dim=1)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError(f"Input must be [B, C, T], got {x.shape}")
        bsz, channels, t = x.shape
        if channels != self.in_channels:
            raise RuntimeError(f"Expected {self.in_channels} channels, got {channels}")

        x0 = x
        x = self.spatial_stem(x)

        w = self.encoder(x)  # [B, N, T']
        if self.use_encoder_relu:
            w = F.relu(w)

        e = self.layer_norm(w)
        e = self.bottleneck(e)

        skip_sum = None
        y = e
        for block in self.separation:
            y, skip = block(y)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        m = self.mask_prelu(skip_sum)
        m = self.mask_norm(m)
        masks = self.gen_masks(m)  # [B, K*N, T']
        masks = masks.view(bsz, self.max_sources, self.N, masks.shape[-1])
        masks = self._apply_mask_activation(masks)

        ests = []
        for k in range(self.max_sources):
            masked = w * masks[:, k]
            est = self.decoder(masked)  # [B, 12, T_est]
            if est.shape[-1] > t:
                est = est[..., :t]
            elif est.shape[-1] < t:
                est = F.pad(est, (0, t - est.shape[-1]))
            ests.append(est)

        ests = torch.stack(ests, dim=1)  # [B, K, 12, T]

        if self.mixture_consistency:
            # Equal correction. Keep disabled by default for variable-source training with silence loss.
            correction = (x0.unsqueeze(1) - ests.sum(dim=1, keepdim=True)) / float(self.max_sources)
            ests = ests + correction

        return ests
