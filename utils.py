import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def calc_gram(x):
    (n, c, h, w) = x.size()
    f = x.view(n, c, w * h)
    f_trans = f.transpose(1, 2)
    gram = f.bmm(f_trans) / (c * h * w)
    return gram


def weights_init(net, scale=1):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=-1, mode='fan_in')
            m.weight.data *= scale  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias.data, 0.0)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size //
                       2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    cs = torch.flatten(cs_map, 2).mean(-1)
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)

    return cs, ssim_per_channel


def calc_ssim(img1, img2, window_size=11):
   """calculate SSIM"""
   (_, channel, _, _) = img1.size()
   window = create_window(window_size, channel)

   if img1.is_cuda:
       window = window.cuda(img1.get_device())
   window = window.type_as(img1)

   ssim_per_channel, _ = _ssim(img1, img2, window, window_size, channel)

   return ssim_per_channel.mean()


def calc_msssim(img1, img2, window_size=11, weights=None):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights).to(img1.device, dtype=img1.dtype)

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
       window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(img1, img2, window, window_size, channel)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in img1.shape[2:]]
            img1 = F.avg_pool2d(img1, kernel_size=2, padding=padding)
            img2 = F.avg_pool2d(img2, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    # (level, batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    return ms_ssim_val.mean().item()


def calc_psnr(img1, img2):
    """calculate PNSR on cuda and cpu: img1 and img2 have range [0, 1]"""
    mse = torch.nn.MSELoss()(img1, img2)
    max_value = torch.max(img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_value / torch.sqrt(mse).item())


def downsample(x):
    """
    :param x: (C, H, W)
    :param noise_sigma: (C, H/2, W/2)
    :return: (4, C, H/2, W/2)
    """
    # x = x[:, :, :x.shape[2] // 2 * 2, :x.shape[3] // 2 * 2]
    N, C, W, H = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

    Cout = 4 * C
    Wout = W // 2
    Hout = H // 2

    if 'cuda' in x.type():
        down_features = torch.cuda.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    else:
        down_features = torch.FloatTensor(N, Cout, Wout, Hout).fill_(0)
    
    for idx in range(4):
        down_features[:, idx:Cout:4, :, :] = x[:, :, idxL[idx][0]::2, idxL[idx][1]::2]

    return down_features


def upsample(x):
    """
    :param x: (n, C, W, H)
    :return: (n, C/4, W*2, H*2)
    """
    N, Cin, Win, Hin = x.size()
    idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    Cout = Cin // 4
    Wout = Win * 2
    Hout = Hin * 2

    up_feature = torch.zeros((N, Cout, Wout, Hout)).type(x.type())
    for idx in range(4):
        up_feature[:, :, idxL[idx][0]::2, idxL[idx][1]::2] = x[:, idx:Cin:4, :, :]

    return up_feature


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
