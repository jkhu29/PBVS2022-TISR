import math
import torch
import torch.nn as nn


class ConvReLU(nn.Module):
    """ConvReLU: conv 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels, out_channels, withbn=True, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(ConvReLU, self).__init__()
        self.withbn = withbn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranReLU(nn.Module):
    """ConvTranReLU: convtran 64 * 3 * 3 + leakyrelu"""
    def __init__(self, in_channels, out_channels, withbn=True, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super(ConvTranReLU, self).__init__()
        self.withbn = withbn
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.withbn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class UpBlock(nn.Module):
    def __init__(self):
        super(UpBlock, self).__init__()
    def forward(self, x):
        pass


class ResBlock(nn.Module):
    """ResBlock"""
    def __init__(self, num_conv=1, channels=64):
        super(ResBlock, self).__init__()

        self.conv_relu = ConvReLU(channels, channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv_relu(x)
        res = x
        x = self.conv(x) + res
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels=64, within=True):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

        self.within = within

    def forward(self, x):
        if self.within:
            output = self.relu(self.in1(self.conv1(x)))
            output = self.in2(self.conv2(output))
        else:
            output = self.conv2(self.relu(self.conv1(x)))
        output = output + x
        return output
