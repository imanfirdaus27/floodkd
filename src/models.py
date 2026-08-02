"""Models.

One U-Net, used three ways:
  * S1 student   in_channels=2   (Sentinel-1 VV, VH)
  * S2 teacher   in_channels=13  (Sentinel-2 all bands)
  * baselines    either of the above, trained alone

forward() can also return the bottleneck feature map, which is what the
pair-wise distillation loss needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, width: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        chans = [width * (2 ** i) for i in range(depth + 1)]     # e.g. 32 64 128 256 512

        self.encoders = nn.ModuleList()
        c_prev = in_channels
        for c in chans[:-1]:
            self.encoders.append(conv_block(c_prev, c))
            c_prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(chans[-2], chans[-1])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.ups.append(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2))
            self.decoders.append(conv_block(chans[i] * 2, chans[i]))

        self.head = nn.Conv2d(chans[0], num_classes, 1)

    def forward(self, x, return_features: bool = False):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        feat = x                                   # bottleneck, used for distillation

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        logits = self.head(x)
        return (logits, feat) if return_features else logits


def build_model(modality: str, cfg) -> UNet:
    in_ch = {'s1': 2, 's2': 13}[modality]
    return UNet(in_ch, num_classes=cfg.num_classes, width=cfg.width, depth=cfg.depth)


class TeacherStudent(nn.Module):
    """Wraps the frozen optical teacher and the trainable SAR student.

    Training reads BOTH modalities. Inference reads Sentinel-1 only, which is
    the entire point of the framework.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, freeze_teacher: bool = True):
        super().__init__()
        self.teacher = teacher
        self.student = student
        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher.eval()

    def forward(self, s1, s2=None):
        s_logits, s_feat = self.student(s1, return_features=True)
        if s2 is None:
            return s_logits, s_feat, None, None       # inference: SAR only
        with torch.no_grad():
            t_logits, t_feat = self.teacher(s2, return_features=True)
        return s_logits, s_feat, t_logits, t_feat

    @torch.no_grad()
    def predict(self, s1):
        return self.student(s1)
