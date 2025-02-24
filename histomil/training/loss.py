import torch
import torch.nn as nn


class FocalBCEWithLogitsLoss(nn.Module):

    def __init__(self, num_label=1, gamma=2):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.num_label = num_label
        self.gamma = gamma

    def forward(self, logits, targets):
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1 - p)
        logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
        loss = logp * ((1 - p)**self.gamma)
        loss = self.num_label * loss.mean()
        return loss