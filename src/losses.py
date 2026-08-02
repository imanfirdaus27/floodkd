"""Losses.

Two groups:
  * segmentation losses  - used by every run (baseline and proposed)
  * distillation losses  - used only by the proposed framework

All of them respect the -1 ignore mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -1


# ------------------------------------------------------------ segmentation
class DiceLoss(nn.Module):
    """Soft Dice on the water class, computed only over valid pixels."""

    def __init__(self, positive: int = 1, eps: float = 1e-6):
        super().__init__()
        self.positive = positive
        self.eps = eps

    def forward(self, logits, target):
        valid = (target != IGNORE_INDEX).float()
        prob = logits.softmax(1)[:, self.positive]
        tgt = (target == self.positive).float()
        prob, tgt = prob * valid, tgt * valid
        inter = (prob * tgt).sum(dim=(1, 2))
        denom = prob.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        return (1 - (2 * inter + self.eps) / (denom + self.eps)).mean()


class SegLoss(nn.Module):
    """Cross-entropy + Dice. class_weights helps because water is the rare class."""

    def __init__(self, dice_weight: float = 0.5, class_weights=None):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        if not (target != IGNORE_INDEX).any():
            return logits.sum() * 0.0
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


# ------------------------------------------------------------ distillation
class ResponseKD(nn.Module):
    """Classic Hinton-style KD on the output logits, masked to valid pixels.

    This is the 'pixel-wise distillation' baseline that Liu et al. (2019) showed
    is NOT enough on its own for segmentation. Keep it as the control.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.t = temperature

    def forward(self, student_logits, teacher_logits, target=None):
        s = F.log_softmax(student_logits / self.t, dim=1)
        t = F.softmax(teacher_logits / self.t, dim=1)
        kl = F.kl_div(s, t, reduction='none').sum(1)          # (B, H, W)
        if target is not None:
            valid = (target != IGNORE_INDEX).float()
            kl = kl * valid
            return (kl.sum() / valid.sum().clamp(min=1)) * (self.t ** 2)
        return kl.mean() * (self.t ** 2)


class PairwiseKD(nn.Module):
    """Pair-wise distillation from Liu et al. (2019).

    Matches the pixel-to-pixel similarity map of teacher and student instead of
    their per-pixel probabilities, so spatial structure is transferred.
    Features are pooled first because a full 512x512 similarity matrix will not
    fit in memory.
    """

    def __init__(self, pool: int = 16):
        super().__init__()
        self.pool = pool

    @staticmethod
    def _similarity(feat):
        b, c, h, w = feat.shape
        f = feat.reshape(b, c, h * w)
        f = F.normalize(f, dim=1)
        return torch.bmm(f.transpose(1, 2), f)                # (B, HW, HW)

    def forward(self, student_feat, teacher_feat):
        s = F.adaptive_avg_pool2d(student_feat, self.pool)
        t = F.adaptive_avg_pool2d(teacher_feat, self.pool)
        return F.mse_loss(self._similarity(s), self._similarity(t))


class ConfidenceGate(nn.Module):
    """Confidence gating, following the idea in Ma et al. (2026).

    An optical teacher looking at a partly clouded scene is not reliable
    everywhere. This down-weights the distillation loss wherever the teacher is
    unsure, so bad supervision does not propagate into the student.
    """

    def __init__(self, threshold: float = 0.7):
        super().__init__()
        self.threshold = threshold

    def forward(self, teacher_logits):
        conf = teacher_logits.softmax(1).max(1).values           # (B, H, W)
        return (conf >= self.threshold).float()


class DistillLoss(nn.Module):
    """Total objective for the proposed framework.

        L = L_seg + alpha * L_response + beta * L_pairwise

    Set alpha/beta to 0 in the config to ablate a component.
    """

    def __init__(self, seg_loss, alpha=1.0, beta=1.0, temperature=4.0,
                 gate_threshold=None, pool=16):
        super().__init__()
        self.seg = seg_loss
        self.response = ResponseKD(temperature)
        self.pairwise = PairwiseKD(pool)
        self.alpha, self.beta = alpha, beta
        self.gate = ConfidenceGate(gate_threshold) if gate_threshold else None

    def forward(self, s_logits, t_logits, target, s_feat=None, t_feat=None):
        out = {'seg': self.seg(s_logits, target)}

        if self.alpha > 0:
            if self.gate is not None:
                mask = self.gate(t_logits)
                masked_target = torch.where(mask.bool(), target,
                                            torch.full_like(target, IGNORE_INDEX))
                out['response'] = self.alpha * self.response(s_logits, t_logits, masked_target)
            else:
                out['response'] = self.alpha * self.response(s_logits, t_logits, target)

        if self.beta > 0 and s_feat is not None and t_feat is not None:
            out['pairwise'] = self.beta * self.pairwise(s_feat, t_feat)

        out['total'] = sum(v for k, v in out.items() if k != 'total')
        return out
