"""Segmentation metrics that ignore the -1 no-data pixels.

Every number the project reports comes from here, so that the ignore mask is
applied in exactly one place and cannot be forgotten in some other script.
"""
from __future__ import annotations

import torch

IGNORE_INDEX = -1


class SegMetrics:
    """Accumulates a confusion matrix over batches, then reports water-class scores."""

    def __init__(self, num_classes: int = 2, device: str = 'cpu'):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.cm = torch.zeros(self.num_classes, self.num_classes,
                              dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """logits (B, C, H, W) raw scores; target (B, H, W) with -1/0/1."""
        pred = logits.argmax(1)
        valid = target != IGNORE_INDEX          # <- the whole point
        p = pred[valid].reshape(-1)
        t = target[valid].reshape(-1)
        idx = t * self.num_classes + p
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.cm += binc.reshape(self.num_classes, self.num_classes).to(self.cm.device)

    def compute(self, positive: int = 1) -> dict:
        """positive=1 means the water class."""
        cm = self.cm.float()
        tp = cm[positive, positive]
        fn = cm[positive].sum() - tp
        fp = cm[:, positive].sum() - tp
        tn = cm.sum() - tp - fn - fp
        eps = 1e-9

        iou = tp / (tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        acc = (tp + tn) / (cm.sum() + eps)

        # mean IoU over both classes, the metric most papers report
        ious = []
        for c in range(self.num_classes):
            t_ = cm[c, c]
            f_n = cm[c].sum() - t_
            f_p = cm[:, c].sum() - t_
            ious.append((t_ / (t_ + f_p + f_n + eps)).item())

        return {
            'iou_water': iou.item(),
            'miou': sum(ious) / len(ious),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'accuracy': acc.item(),
            'n_valid_px': int(cm.sum().item()),
        }

    def __str__(self) -> str:
        m = self.compute()
        return (f"IoU(water) {m['iou_water']:.4f} | mIoU {m['miou']:.4f} | "
                f"F1 {m['f1']:.4f} | P {m['precision']:.4f} | R {m['recall']:.4f} | "
                f"Acc {m['accuracy']:.4f}")
