import torch
import torch.nn as nn
import torch.nn.functional as F


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self,t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-8

    def forward(self, x, y):
        diff = F.pairwise_distance(x, y, p=1)
        error = torch.sqrt(diff ** 2 + self.eps)
        loss = torch.sum(error)
        return loss
