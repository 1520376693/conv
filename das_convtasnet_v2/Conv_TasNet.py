import torch
import torch.nn as nn


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization for 3D tensors [B, C, T]."""

    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, 1))
            self.bias = nn.Parameter(torch.zeros(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError(f"GlobalLayerNorm expects [B, C, T], got {x.shape}")
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, dim=(1, 2), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = self.weight * x + self.bias
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Channel-wise LayerNorm for 3D tensors [B, C, T]."""

    def __init__(self, dim, elementwise_affine=True):
        super().__init__(dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError(f"CumulativeLayerNorm expects [B, C, T], got {x.shape}")
        x = torch.transpose(x, 1, 2)  # [B, T, C]
        x = super().forward(x)
        x = torch.transpose(x, 1, 2)  # [B, C, T]
        return x


def select_norm(norm, dim):
    norm = norm.lower()
    if norm == "gln":
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "bn":
        return nn.BatchNorm1d(dim)
    raise ValueError("norm must be one of: gln, cln, bn")


class Conv1D(nn.Conv1d):
    """Conv1d wrapper. Supports [B, T] or [B, C, T]."""

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError(f"Conv1D expects 2D/3D tensor, got {x.shape}")
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """ConvTranspose1d wrapper. Supports [B, T] or [B, C, T]."""

    def forward(self, x, squeeze=False, output_size=None):
        if x.dim() not in [2, 3]:
            raise RuntimeError(f"ConvTrans1D expects 2D/3D tensor, got {x.shape}")
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1), output_size=output_size)
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(nn.Module):
    """
    Backward-compatible residual TCN block.
    Kept so old checkpoints/scripts importing this class still work.
    New separator uses TemporalBlockWithSkip below.
    """

    def __init__(
        self,
        in_channels=256,
        out_channels=512,
        kernel_size=3,
        dilation=1,
        norm="gln",
        causal=False,
    ):
        super().__init__()
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.prelu_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else dilation * (kernel_size - 1)
        self.dwconv = Conv1D(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            padding=self.pad,
            dilation=dilation,
        )
        self.prelu_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        self.sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.prelu_1(y)
        y = self.norm_1(y)
        y = self.dwconv(y)
        if self.causal and self.pad > 0:
            y = y[:, :, :-self.pad]
        y = self.prelu_2(y)
        y = self.norm_2(y)
        y = self.sc_conv(y)
        return x + y


class TemporalBlockWithSkip(nn.Module):
    """
    Conv-TasNet TCN block with residual and skip outputs.

    Compared with a pure residual stack, the skip path gives the mask generator
    direct access to multi-scale features from every dilation layer, which usually
    converges faster and separates overlapped vibration components more stably.
    """

    def __init__(
        self,
        channels=256,
        hidden_channels=512,
        kernel_size=3,
        dilation=1,
        norm="gln",
        causal=False,
        dropout=0.0,
    ):
        super().__init__()
        self.causal = causal
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else dilation * (kernel_size - 1)

        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1, bias=False),
            nn.PReLU(),
            select_norm(norm, hidden_channels),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                padding=self.pad,
                dilation=dilation,
                groups=hidden_channels,
                bias=False,
            ),
            nn.PReLU(),
            select_norm(norm, hidden_channels),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.res_out = nn.Conv1d(hidden_channels, channels, 1, bias=True)
        self.skip_out = nn.Conv1d(hidden_channels, channels, 1, bias=True)

    def forward(self, x):
        y = self.net(x)
        if self.causal and self.pad > 0:
            y = y[:, :, :-self.pad]
        residual = self.res_out(y)
        skip = self.skip_out(y)
        return x + residual, skip


def check_parameters(net):
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 1e6
