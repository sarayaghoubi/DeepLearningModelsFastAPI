import torch
import torch.nn as nn
import torch.nn.functional as functional


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels=None, k=(4, 1)):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=k),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=k),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with max pool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, k: (int, int)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, None, k)
        )

    def forward(self, x):
        return self.max_pool_conv(x)


class Up(nn.Module):
    """Up scaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bi_linear=True, k=(2, 2)):
        super().__init__()

        # if bi_linear, use the normal convolutions to reduce the number of channels
        if bi_linear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, kernel_size=k)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=k)
            self.conv = DoubleConv(in_channels, out_channels, None, k=k)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, k_size=(4, 2), nodes=1600):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.lin1 = nn.Sequential(
            nn.Linear(nodes, 1028),
            nn.Dropout(0.2),
            nn.LeakyReLU()
        )
        self.lin2 = nn.Sequential(
            nn.Linear(1028, 512),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
        )
        self.final = nn.Sequential(
            nn.Linear(512, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.final(x)
        return torch.squeeze(x)
