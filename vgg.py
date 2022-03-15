import torch
import torch.nn as nn
import torchvision
from collections import namedtuple


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        vgg_pretrained_features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class GAN_D(nn.Module):
    """GAN_D: VGG19"""
    def __init__(self):
        super(GAN_D, self).__init__()
        self.net_d = [nn.Conv2d(1, 64, kernel_size=3, padding=1)]
        self.net_d.extend([nn.LeakyReLU(0.2)])
        self._conv_block(64, 64, with_stride=True)
        self._conv_block(64, 128)
        self._conv_block(128, 128, with_stride=True)
        self._conv_block(128, 256)
        self._conv_block(256, 256, with_stride=True)
        self._conv_block(256, 512)
        self._conv_block(512, 512, with_stride=True)
        self.net_d.extend([nn.AdaptiveAvgPool2d(1)])
        self.net_d.extend([nn.Conv2d(512, 1024, kernel_size=1)])
        self.net_d.extend([nn.LeakyReLU(0.2)])
        self.net_d.extend([nn.Conv2d(1024, 1, kernel_size=1)])
        self.net_d = nn.Sequential(*self.net_d)

    def _conv_block(self, in_channels, out_channels, with_batch=True, with_stride=False):
        if with_stride:
            self.net_d.extend([nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)])
        else:
            self.net_d.extend([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)])

        if with_batch:
            self.net_d.extend([nn.BatchNorm2d(out_channels)])
        self.net_d.extend([nn.LeakyReLU(0.2)])

    def forward(self, x):
        batch_size = x.size(0)
        out = self.net_d(x).view(batch_size)
        return out
