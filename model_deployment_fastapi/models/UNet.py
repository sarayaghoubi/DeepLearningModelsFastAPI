from bricks import *


class UNet30(nn.Module):
    def __init__(self, n_channels, n_classes, bi_linear=False):
        super(UNet30, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bi_linear = bi_linear
        self.inc = DoubleConv(n_channels, 128, None, k=(4,2))
        self.down1 = Down(128, 256,k=(4,1))
        self.down2 = Down(256, 512,k=(4,1))
        self.down3 = Down(512, 1024,k=(4,1))
        factor = 2 if bi_linear else 1
        self.down4 = Down(1024, 2048 // factor, k=(4,1))
        self.up1 = Up(2048, 1024 // factor, bi_linear, k=(4,2))
        self.up2 = Up(1024, 512 // factor, bi_linear, k=(4,2))
        self.up3 = Up(512, 256 // factor, bi_linear, k=(4,2))
        self.up4 = Up(256, 128, bi_linear, k=(2,2))
        self.outc = OutConv(128, n_classes, (4,2), nodes=37)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logit = self.outc(x)
        return logit
