"""Sen1Floods11 data: download + PyTorch Dataset.

The official hand-labelled split files list pairs like
    Ghana_5079_S1Hand.tif,Ghana_5079_LabelHand.tif
so the chip id is everything before the first '_S1Hand'.
"""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

GCS = 'https://storage.googleapis.com/sen1floods11/v1.1'
DATA_URL = f'{GCS}/data/flood_events/HandLabeled'
SPLIT_URL = f'{GCS}/splits/flood_handlabeled'

SPLITS = {
    'train': 'flood_train_data.csv',
    'val': 'flood_valid_data.csv',
    'test': 'flood_test_data.csv',
    'bolivia': 'flood_bolivia_data.csv',   # held-out event, never train on this
}

# Sentinel-1 is in dB. Real values sit roughly in [-50, 1].
S1_MIN, S1_MAX = -50.0, 1.0
# Sentinel-2 L1C is TOA reflectance scaled by 10000.
S2_SCALE = 10000.0

IGNORE_INDEX = -1          # label value that must never contribute to loss or metrics


# --------------------------------------------------------------------- download
def _get(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def read_split(split: str, root: Path) -> list[str]:
    """Return the list of chip ids for a split, downloading the CSV if needed."""
    if split not in SPLITS:
        raise ValueError(f'unknown split {split!r}, expected one of {list(SPLITS)}')
    csv_path = root / 'splits' / SPLITS[split]
    _get(f'{SPLIT_URL}/{SPLITS[split]}', csv_path)

    chips = []
    with open(csv_path, newline='') as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                chips.append(row[0].strip().replace('_S1Hand.tif', ''))
    return chips


def download_split(split: str, root: Path, modalities=('S1Hand', 'S2Hand', 'LabelHand'),
                   verbose: bool = True) -> list[str]:
    """Download every chip in a split. Safe to re-run: existing files are skipped."""
    chips = read_split(split, root)
    for i, chip in enumerate(chips, 1):
        for m in modalities:
            name = f'{chip}_{m}.tif'
            _get(f'{DATA_URL}/{m}/{name}', root / m / name)
        if verbose and (i % 25 == 0 or i == len(chips)):
            print(f'  {split:<8} {i:>4}/{len(chips)} chips')
    return chips


# ---------------------------------------------------------------------- dataset
class Sen1Floods11(Dataset):
    """One item = one 512x512 chip.

    Returns a dict:
        s1    float32 (2, H, W)   scaled to [0, 1]
        s2    float32 (13, H, W)  scaled to [0, 1]
        label int64   (H, W)      values -1 / 0 / 1
        chip  str
    Which tensors are actually loaded depends on `modality`, so an S1-only run
    does not pay the cost of reading 13-band optical files.
    """

    def __init__(self, root, split='train', modality='both', transform=None):
        self.root = Path(root)
        self.split = split
        self.modality = modality
        self.transform = transform
        self.chips = read_split(split, self.root)
        self.need_s1 = modality in ('s1', 'both')
        self.need_s2 = modality in ('s2', 'both')

    def __len__(self) -> int:
        return len(self.chips)

    def _read(self, chip: str, layer: str) -> np.ndarray:
        path = self.root / layer / f'{chip}_{layer}.tif'
        if not path.exists():
            raise FileNotFoundError(
                f'{path} missing. Run:  python run.py download --split {self.split}')
        with rasterio.open(path) as src:
            return src.read()

    def __getitem__(self, idx: int) -> dict:
        chip = self.chips[idx]
        item = {'chip': chip}

        if self.need_s1:
            s1 = self._read(chip, 'S1Hand').astype(np.float32)
            s1 = np.nan_to_num(s1, nan=S1_MIN, posinf=S1_MAX, neginf=S1_MIN)
            s1 = (np.clip(s1, S1_MIN, S1_MAX) - S1_MIN) / (S1_MAX - S1_MIN)
            item['s1'] = torch.from_numpy(s1)

        if self.need_s2:
            s2 = self._read(chip, 'S2Hand').astype(np.float32) / S2_SCALE
            s2 = np.clip(np.nan_to_num(s2), 0.0, 1.0)
            item['s2'] = torch.from_numpy(s2)

        label = self._read(chip, 'LabelHand')[0].astype(np.int64)
        item['label'] = torch.from_numpy(label)

        if self.transform is not None:
            item = self.transform(item)
        return item


def class_balance(ds: Sen1Floods11) -> dict:
    """Count label pixels across a dataset. Useful for setting class weights."""
    counts = {-1: 0, 0: 0, 1: 0}
    for i in range(len(ds)):
        lab = ds[i]['label'].numpy()
        v, c = np.unique(lab, return_counts=True)
        for vi, ci in zip(v, c):
            counts[int(vi)] = counts.get(int(vi), 0) + int(ci)
    total = sum(counts.values())
    return {k: (v, 100 * v / total) for k, v in counts.items()}
