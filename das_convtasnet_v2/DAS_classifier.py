import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class Residual2DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1), dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(9, 3), stride=stride, padding=(4, 1), bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(9, 3), stride=(1, 1), padding=(4, 1), bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == (1, 1)
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        y = self.drop(y)
        return self.act(y + self.shortcut(x))


class DASResNetClassifier(nn.Module):
    """
    2D residual CNN classifier for DAS samples.

    输入: [B, 1, T, 12]
    输出: [B, 6]

    设计要点：
    - 长时间维用大卷积核/步长快速降采样；
    - 空间维保留12个邻近通道的信息；
    - residual + SE提升稳定性和泛化；
    - 训练时配合DASDataLoaders中的噪声/平移/遮挡增强。
    """

    def __init__(self, num_classes=6, dropout=0.35, base_channels=32):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=(101, 3), stride=(10, 1), padding=(50, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
        )
        self.stage1 = nn.Sequential(
            Residual2DBlock(c, c, stride=(1, 1), dropout=0.05),
            Residual2DBlock(c, c, stride=(1, 1), dropout=0.05),
        )
        self.stage2 = nn.Sequential(
            Residual2DBlock(c, c * 2, stride=(2, 1), dropout=0.05),
            Residual2DBlock(c * 2, c * 2, stride=(1, 1), dropout=0.05),
        )
        self.stage3 = nn.Sequential(
            Residual2DBlock(c * 2, c * 4, stride=(2, 1), dropout=0.10),
            Residual2DBlock(c * 4, c * 4, stride=(1, 1), dropout=0.10),
            SEBlock(c * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * 4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


# Keep the old class name so existing train/eval scripts do not need to change.
DASBaselineCNN = DASResNetClassifier
