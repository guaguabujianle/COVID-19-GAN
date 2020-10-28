import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        out = self.inc(x)

        return out

class Conv1x1Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        out = self.inc(x)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = False
        if in_channels != out_channels:
            self.downsample = True

        self.inc = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.InstanceNorm2d(out_channels, affine=False),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.InstanceNorm2d(out_channels, affine=False),
        )

        self.conv1x1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        nn.InstanceNorm2d(out_channels, affine=False)
        )

    def forward(self, x):
        if self.downsample:
            skip = self.conv1x1(x)
        else:
            skip = x
        x_inc = self.inc(x)
        out = F.leaky_relu(skip + x_inc, 0.2, True)

        # print(skip.size())
        # print(x_inc.size())
        # print(out.size())

        return out

class ResBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = False
        if in_channels != out_channels:
            self.downsample = True

        self.inc = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.InstanceNorm2d(out_channels, affine=False),
        nn.ReLU(True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.InstanceNorm2d(out_channels, affine=False),
        )

        self.conv1x1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
        nn.InstanceNorm2d(out_channels, affine=False),
        )

    def forward(self, x):
        if self.downsample:
            skip = self.conv1x1(x)
        else:
            skip = x
        x_inc = self.inc(x)
        out = F.relu(skip + x_inc, True)

        # print(skip.size())
        # print(x_inc.size())
        # print(out.size())

        return out


class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 2)):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, affine=False),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        out = self.inc(x)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = ConvDown(in_channels, in_channels)
        self.res_block = ResBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        out = self.res_block(x)

        return out

class Up(nn.Module):
    def __init__(self, up_channels, skip_channels, out_channels):
        super().__init__()
        in_channels = up_channels + skip_channels
        self.up = nn.Sequential(nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2),
        nn.InstanceNorm2d(up_channels, affine=False),
        nn.ReLU(inplace=False))
        self.res_block = ResBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # up sample
        x1 = self.up(x1)
        # padding if needed
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        p = (0, diff_w, 0, diff_h)
        x1 = F.pad(x1, p)
        x = torch.cat([x2, x1], dim=1)

        out = self.res_block(x)

        return out

 
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.outconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.outconv(x))


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        channels = [32, 64, 96, 128, 256, 384, 512]
        self.inc = Conv3x3Layer(n_channels, channels[0])
        self.encoder0 = ResBlock(channels[0], channels[1])
        self.encoder1 = Down(channels[1], channels[2])
        self.encoder2 = Down(channels[2], channels[3])
        self.encoder3 = Down(channels[3], channels[4])
        self.encoder4 = Down(channels[4], channels[5])
        self.encoder5 = Down(channels[5], channels[6])

        self.decoder4 = Up(channels[6], channels[5], channels[5])
        self.decoder3 = Up(channels[5], channels[4], channels[4])
        self.decoder2 = Up(channels[4], channels[3], channels[3])
        self.decoder1 = Up(channels[3], channels[2], channels[2])
        self.decoder0 = Up(channels[2], channels[1], channels[1])    

        self.out_conv = OutConv(channels[1], n_classes)


    def forward(self, x):
        x = self.inc(x)

        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        d4 = self.decoder4(e5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        d0 = self.decoder0(d1, e0)

        out = self.out_conv(d0)

        # print("x:", x.size())
        # print("e0:", e0.size())
        # print("e1:", e1.size())
        # print("e2:", e2.size())
        # print("e3:", e3.size())
        # print("e4:", e4.size())
        # print("e5:", e5.size())
        # print("d4:", d4.size())
        # print("d3:", d3.size())
        # print("d2:", d2.size())
        # print("d1:", d1.size())
        # print("d0:", d0.size())
        # print("out:", out.size())

        return out


class NonLocalUnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        channels = [32, 64, 96, 128, 256, 384, 512]
        self.inc = Conv3x3Layer(n_channels, channels[0])
        self.encoder0 = ResBlock(channels[0], channels[1])
        self.encoder1 = Down(channels[1], channels[2])
        self.encoder2 = Down(channels[2], channels[3])
        self.encoder3 = Down(channels[3], channels[4])
        self.encoder4 = Down(channels[4], channels[5])
        self.encoder5 = Down(channels[5], channels[6])

        self.layer_att3 = NonLocalBlock(channels[4], channels[4], sub_sample=False, in_layer=True)
        self.layer_att4 = NonLocalBlock(channels[5], channels[5], sub_sample=False, in_layer=True)
        self.layer_att5 = NonLocalBlock(channels[6], channels[6], sub_sample=False, in_layer=True)

        self.decoder4 = Up(channels[6], channels[5], channels[5])
        self.decoder3 = Up(channels[5], channels[4], channels[4])
        self.decoder2 = Up(channels[4], channels[3], channels[3])
        self.decoder1 = Up(channels[3], channels[2], channels[2])
        self.decoder0 = Up(channels[2], channels[1], channels[1])    

        self.out_conv = OutConv(channels[1], n_classes)


    def forward(self, x):
        x = self.inc(x)

        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e3 = self.layer_att3(e3)
        e4 = self.encoder4(e3)
        e4 = self.layer_att4(e4)
        e5 = self.encoder5(e4)
        e5 = self.layer_att5(e5)

        d4 = self.decoder4(e5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        d0 = self.decoder0(d1, e0)

        out = self.out_conv(d0)

        # print("x:", x.size())
        # print("e0:", e0.size())
        # print("e1:", e1.size())
        # print("e2:", e2.size())
        # print("e3:", e3.size())
        # print("e4:", e4.size())
        # print("e5:", e5.size())
        # print("d4:", d4.size())
        # print("d3:", d3.size())
        # print("d2:", d2.size())
        # print("d1:", d1.size())
        # print("d0:", d0.size())
        # print("out:", out.size())

        return out



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = NonLocalUnet(1, 1)
    x = torch.rand(1, 1, 256, 256)
    res = net(x)

