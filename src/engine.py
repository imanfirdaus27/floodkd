"""Training / evaluation loops, checkpointing, logging, reproducibility."""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .metrics import SegMetrics


def set_seed(seed: int) -> None:
    """Same seed everywhere. Without this you cannot compare two runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    """Writes one CSV row per epoch, plus the config as JSON. No hidden state."""

    def __init__(self, out_dir: Path, cfg=None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / 'log.csv'
        self.fields = None
        if cfg is not None:
            (self.dir / 'config.json').write_text(json.dumps(vars(cfg), indent=2, default=str))

    def log(self, row: dict) -> None:
        if self.fields is None:
            self.fields = list(row)
            with open(self.csv_path, 'w', newline='') as f:
                csv.DictWriter(f, self.fields).writeheader()
        with open(self.csv_path, 'a', newline='') as f:
            csv.DictWriter(f, self.fields).writerow({k: row.get(k) for k in self.fields})


def save_ckpt(path: Path, model, optimizer, epoch: int, best: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'best': best}, path)


def load_ckpt(path: Path, model, optimizer=None):
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model'])
    if optimizer is not None and 'optimizer' in ck:
        optimizer.load_state_dict(ck['optimizer'])
    return ck.get('epoch', 0), ck.get('best', 0.0)


# ------------------------------------------------------------------ baseline
def train_one_epoch(model, loader, loss_fn, optimizer, device, modality, scaler=None):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x = batch[modality].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.split(':')[0], dtype=torch.float16):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, modality, loss_fn=None):
    model.eval()
    metrics = SegMetrics(device=device)
    total, n = 0.0, 0
    for batch in loader:
        x = batch[modality].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        logits = model(x)
        if loss_fn is not None:
            total += loss_fn(logits, y).item() * x.size(0)
            n += x.size(0)
        metrics.update(logits, y)
    out = metrics.compute()
    out['loss'] = total / max(n, 1) if loss_fn is not None else None
    return out


# --------------------------------------------------------------- distillation
def train_one_epoch_distill(ts_model, loader, distill_loss, optimizer, device, scaler=None):
    ts_model.student.train()
    ts_model.teacher.eval()
    sums, n = {}, 0
    for batch in loader:
        s1 = batch['s1'].to(device, non_blocking=True)
        s2 = batch['s2'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        s_logits, s_feat, t_logits, t_feat = ts_model(s1, s2)
        parts = distill_loss(s_logits, t_logits, y, s_feat, t_feat)

        if scaler is not None:
            scaler.scale(parts['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            parts['total'].backward()
            optimizer.step()

        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v) * s1.size(0)
        n += s1.size(0)
    return {k: v / max(n, 1) for k, v in sums.items()}


def fit(model, train_loader, val_loader, loss_fn, optimizer, device, cfg,
        out_dir: Path, modality: str, scheduler=None, distill_loss=None):
    """Shared training driver for both the baseline and the distillation runs."""
    logger = Logger(out_dir, cfg)
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.startswith('cuda')) else None
    best = -1.0
    is_distill = distill_loss is not None

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        if is_distill:
            tr = train_one_epoch_distill(model, train_loader, distill_loss,
                                         optimizer, device, scaler)
            train_loss = tr['total']
            eval_model = model.student
        else:
            train_loss = train_one_epoch(model, train_loader, loss_fn,
                                         optimizer, device, modality, scaler)
            eval_model = model

        val = evaluate(eval_model, val_loader, device, 's1' if is_distill else modality, loss_fn)
        if scheduler is not None:
            scheduler.step()

        row = {'epoch': epoch, 'train_loss': round(train_loss, 5),
               'val_loss': round(val['loss'], 5) if val['loss'] is not None else None,
               'val_iou_water': round(val['iou_water'], 5),
               'val_miou': round(val['miou'], 5),
               'val_f1': round(val['f1'], 5),
               'secs': round(time.time() - t0, 1)}
        logger.log(row)
        print(f"epoch {epoch:>3}/{cfg.epochs}  loss {train_loss:.4f}  "
              f"val IoU(water) {val['iou_water']:.4f}  F1 {val['f1']:.4f}  "
              f"({row['secs']}s)")

        save_ckpt(out_dir / 'last.pt', eval_model, optimizer, epoch, best)
        if val['iou_water'] > best:
            best = val['iou_water']
            save_ckpt(out_dir / 'best.pt', eval_model, optimizer, epoch, best)
            print(f'         new best IoU {best:.4f} -> best.pt')

    return best
