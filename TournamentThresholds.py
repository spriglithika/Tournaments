import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


class _BaseThreshold(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = int(num_classes)
        perms = torch.tensor(list(combinations(range(self.num_classes), 2)), dtype=torch.long)
        # register permutations as a buffer so they move with the module
        self.register_buffer('perms', perms)


class NaiveThresholding(_BaseThreshold):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def forward(self, x):
        # x: (B, E) floats in [0,1]
        # binarize around 0.5
        binarized = (x > 0.5).to(torch.long)
        B, E = binarized.shape

        left = self.perms[:, 0]
        right = self.perms[:, 1]

        left_exp = left.unsqueeze(0).expand(B, -1)
        right_exp = right.unsqueeze(0).expand(B, -1)

        selected = torch.where(binarized == 0, left_exp.to(binarized.device), right_exp.to(binarized.device))

        counts = F.one_hot(selected, num_classes=self.num_classes).sum(dim=1).to(torch.long)
        preds = counts.to(torch.float32).argmax(dim=1)
        return F.one_hot(preds, num_classes=self.num_classes)


class CenterThresholding(_BaseThreshold):
    def __init__(self, num_classes, alpha=0.1):
        super().__init__(num_classes)
        self.alpha = float(alpha)

    def forward(self, x):
        # x: (B, E) floats in [0,1]
        B, E = x.shape
        left = self.perms[:, 0]
        right = self.perms[:, 1]
        # center index should be K (with one-hot num_classes K+1 we have indices 0..K)
        center = torch.full_like(left, fill_value=self.num_classes, dtype=torch.long)

        left_exp = left.unsqueeze(0).expand(B, -1).to(x.device)
        right_exp = right.unsqueeze(0).expand(B, -1).to(x.device)
        center_exp = center.unsqueeze(0).expand(B, -1).to(x.device)

        # create trinary: -1,0,1
        below = x <= (0.5 - self.alpha)
        above = x >= (0.5 + self.alpha)
        trinary = torch.where(below, -1, torch.where(above, 1, 0)).to(torch.long)

        selected = torch.where(trinary == -1, left_exp, torch.where(trinary == 0, center_exp, right_exp))

        # counts over K+1 classes, discard center class counts (last column)
        counts = F.one_hot(selected, num_classes=self.num_classes + 1).sum(dim=1).to(torch.long)[:, :self.num_classes]
        preds = counts.to(torch.float32).argmax(dim=1)
        return F.one_hot(preds, num_classes=self.num_classes)


class BernoulliThresholding(_BaseThreshold):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def forward(self, x):
        # x in [0,1], sample bernoulli
        bits = torch.bernoulli(x).to(torch.long)
        B, E = bits.shape
        left = self.perms[:, 0]
        right = self.perms[:, 1]
        left_exp = left.unsqueeze(0).expand(B, -1).to(bits.device)
        right_exp = right.unsqueeze(0).expand(B, -1).to(bits.device)
        selected = torch.where(bits == 0, left_exp, right_exp)
        counts = F.one_hot(selected, num_classes=self.num_classes).sum(dim=1).to(torch.long)
        preds = counts.to(torch.float32).argmax(dim=1)
        return F.one_hot(preds, num_classes=self.num_classes)


class SeparateConfidence(_BaseThreshold):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def forward(self, x):
        B, E = x.shape
        left = self.perms[:, 0]
        right = self.perms[:, 1]

        left_idx = left.unsqueeze(0).expand(B, -1).to(x.device)
        right_idx = right.unsqueeze(0).expand(B, -1).to(x.device)
        # counts via scatter_add for efficiency
        counts = torch.zeros((B, self.num_classes), dtype=torch.float32, device=x.device)
        # Build low/high confidences from single x input as original logic
        x_temp = x.clone()
        x_temp = x_temp - 0.5
        x_temp = x_temp * 2.0
        x_low = torch.clamp(x_temp, max=0).abs()
        x_high = torch.clamp(x_temp, min=0)
        # left contributions (x_low shape (B,E), left_idx shape (B,E))
        counts.scatter_add_(1, left_idx, x_low)
        # right contributions
        counts.scatter_add_(1, right_idx, x_high)

        preds = counts.argmax(dim=1)
        return F.one_hot(preds, num_classes=self.num_classes)


class SingleConfidence(_BaseThreshold):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def forward(self, x):
        # x: (B, E)
        B, E = x.shape
        left = self.perms[:, 0]
        right = self.perms[:, 1]
        left_idx = left.unsqueeze(0).expand(B, -1).to(x.device)
        right_idx = right.unsqueeze(0).expand(B, -1).to(x.device)


        counts = torch.zeros((B, self.num_classes), dtype=torch.float32, device=x.device)
        # add high contributions
        counts.scatter_add_(1, right_idx, (1-x) )
        # add low contributions (1 - confidence)
        counts.scatter_add_(1, left_idx, x)

        preds = counts.argmax(dim=1)
        return F.one_hot(preds, num_classes=self.num_classes)

