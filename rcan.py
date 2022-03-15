import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.attention_modules import ChannelAttention
from lib.common import ConvReLU
from rfdn import ESA


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)

    
class ResidualDenseBlock_5C(nn.Module):
    '''Residual Dense Block '''
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

    
class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.c2 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c3 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.c5 = nn.Conv2d(in_channels, in_channels, 1)
        self.cca = ChannelAttention(self.distilled_channels * 4)

    def forward(self, x):
        out_c1 = self.act(self.c1(x))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + x
        return out_fused


class SeparableConv(nn.Module):
    def __init__(self, n_feats=50, k=3):
        super(SeparableConv, self).__init__()
        self.separable_conv = nn.Sequential(
            nn.Conv2d(n_feats, 2*n_feats, k, 1, (k-1)//2, groups=n_feats),
            nn.ReLU(True),
            nn.Conv2d(2*n_feats, n_feats, 1, 1, 0),
        )
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.separable_conv(x)
        out += x
        out = self.act(out)

        return out

            
class Cell(nn.Module):
    def __init__(self, n_feats=50):
        super(Cell, self).__init__()
        self.conv1x1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.separable_conv7x7 = SeparableConv(n_feats, k=7)
        self.separable_conv5x5 = SeparableConv(n_feats, k=5)
        self.fuse = nn.Conv2d(n_feats*2, n_feats, 1, 1, 0)

        self.esa = ESA(n_feats)

        self.branch = nn.ModuleList([nn.Conv2d(n_feats, n_feats//2, 1, 1, 0) for _ in range(4)])

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.separable_conv7x7(out1)
        out3 = self.separable_conv5x5(out2)

        # fuse [x, out1, out2, out3]
        out = self.fuse(torch.cat([self.branch[0](x), self.branch[1](out1), self.branch[2](out2), self.branch[3](out3)] ,dim=1))
        out = self.esa(out)
        out += x

        return out


class RCAN(nn.Module):
    def __init__(
        self, scale=2, num_features=64, num_rg=5, num_rcab=10, reduction=8, 
        block: nn.Module = RG
    ):
        super(RCAN, self).__init__()
        self.scale = scale

        self.conv1 = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[block(num_features, num_rcab, reduction) for _ in range(num_rg)])
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            ConvReLU(num_features, 1 * (scale ** 2)),
            nn.PixelShuffle(scale),
        )
        self.conv3 = nn.Conv2d(num_features, 1, 1)

    def forward(self, x, scale=0.1):
        out = self.conv1(x)
        out1 = self.conv2(self.rgs(out))
        out = out + out1
        if self.scale != 1:
            res = F.interpolate(out, scale_factor=self.scale)
            out = self.upsample(out)
            out = self.conv3(out + res * scale)
        else:
            out = self.conv3(out)
        return out

    
def make_cleaning_net():
    return RCAN(scale=1, num_rcab=5)


def make_sr_net(scale=4):
    return RCAN(scale=scale, num_rcab=10)


if __name__ == "__main__":
    model = RCAN().cuda()
    dummy_input = torch.rand((2, 1, 32, 32)).cuda()
    dummy_output = model(dummy_input)
    print(dummy_output.shape)
