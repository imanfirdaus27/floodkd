"""Config. Every number the experiment depends on lives here or in a YAML file.

Nothing in the codebase should contain a magic number. If a reviewer asks
"what learning rate did you use for that table?", the answer is in the
config.json that gets written next to the checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class Config:
    # data
    root: str = 'data'
    modality: str = 's1'            # s1 | s2 | both
    batch_size: int = 4
    num_workers: int = 2

    # model
    num_classes: int = 2
    width: int = 32
    depth: int = 4

    # optimisation
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    class_weights: tuple = (1.0, 5.0)   # water is rare, so weight it up
    amp: bool = True                    # mixed precision, ignored on CPU

    # distillation (only used by `train-distill`)
    alpha: float = 1.0                  # weight on response KD
    beta: float = 1.0                   # weight on pair-wise KD
    temperature: float = 4.0
    gate_threshold: float = 0.0         # 0 disables confidence gating
    teacher_ckpt: str = 'runs/teacher_s2/best.pt'

    # bookkeeping
    seed: int = 42
    out_dir: str = 'runs/exp'
    device: str = 'auto'                # auto | cpu | cuda

    def resolve_device(self) -> str:
        if self.device != 'auto':
            return self.device
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | None = None, **overrides) -> 'Config':
        data = {}
        if path and Path(path).exists():
            import yaml
            data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        data.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(data) - known
        if unknown:
            raise ValueError(f'unknown config keys: {sorted(unknown)}')
        return cls(**data)
