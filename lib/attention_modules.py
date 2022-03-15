import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class ChannelAttention(nn.Module):
    def __init__(self, num_features: int = 64, reduction: int = 4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(self.avg_pool(x))


class ChannelAttentionWeight(ChannelAttention):
    def forward(self, x):
        return self.conv(self.avg_pool(x))


class NonLocalAttention(nn.Module):
    def __init__(self, channels: int = 64):
        super(NonLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_embed1 = self.relu1(self.conv1(x))
        x_embed2 = self.relu2(self.conv2(x))
        x_assembly = self.relu3(self.conv3(x))

        n, c, h, w = x_embed1.shape
        x_embed1 = x_embed1.permute(0, 2, 3, 1).view(n, h * w, c)
        x_embed2 = x_embed2.view(n, c, h * w)
        score = torch.matmul(x_embed1, x_embed2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(n, -1, h * w).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly).permute(0, 2, 1).view(n, -1, h, w)

        return score, x_final


class PixelAttention(nn.Module):
    def __init__(self, channels: int = 64):
        super(PixelAttention, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = self.sigmoid(x)
        return x.mul_(res)


class PSABlock(nn.Module):
    def __init__(self, channels: int = 64, stride: int = 1, kernel_sizes: List = [3, 5, 7, 9], groups: List = [1, 4, 8, 16]):
        super(PSABlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2, stride=stride, groups=groups[0])
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2, stride=stride, groups=groups[0])
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2, stride=stride, groups=groups[0])
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2, stride=stride, groups=groups[0])
        self.ca_weight = ChannelAttentionWeight(channels // 4)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        features = torch.cat([x0, x1, x2, x3], dim=1)
        features = features.view(x.shape[0], 4, x.shape[1] // 4, x.shape[2], x.shape[3])

        x0_ca = self.ca_weight(x0)
        x1_ca = self.ca_weight(x1)
        x2_ca = self.ca_weight(x2)
        x3_ca = self.ca_weight(x3)

        x_ca = torch.cat([x0_ca, x1_ca, x2_ca, x3_ca], dim=1).view(x.shape[0], 4, self.shape[1] // 4, 1, 1)
        weight = torch.softmax(x_ca, dim=1)
        features *= weight

        # TODO(JiaKui Hu)
        for i in range(4):
            f = features[:, i, :, :]
            if i == 0:
                out = f
            else:
                out = torch.cat((f, out), 1)
        return out
