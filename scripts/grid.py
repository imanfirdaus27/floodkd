"""Show many chips at once.

Lives in scripts/. Run it FROM THE PROJECT ROOT (the folder with run.py):

    python scripts/grid.py                     12 chips from val, most water first
    python scripts/grid.py --n 20              more chips
    python scripts/grid.py --split train       a different split
    python scripts/grid.py --order random      instead of most-water-first
    python scripts/grid.py --order least       the hard ones: barely any water

Only chips already downloaded are used, so run `download` first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))   # so `src` is importable

from src.data import read_split, IGNORE_INDEX


def stretch(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--order', default='water',
                    choices=['water', 'least', 'random', 'file'])
    ap.add_argument('--out', default='figures/grid.png')
    args = ap.parse_args()

    root = Path(args.root)
    chips = read_split(args.split, root)

    # keep only what is on disk, and measure how much water each one has
    rows = []
    for c in chips:
        lab_path = root / 'LabelHand' / f'{c}_LabelHand.tif'
        s1_path = root / 'S1Hand' / f'{c}_S1Hand.tif'
        if not (lab_path.exists() and s1_path.exists()):
            continue
        with rasterio.open(lab_path) as src:
            lab = src.read(1)
        valid = lab != IGNORE_INDEX
        if not valid.any():
            continue                       # fully clouded, nothing to show
        water = 100 * (lab == 1).sum() / valid.sum()
        rows.append((c, water, valid.mean() * 100))

    if not rows:
        raise SystemExit(f'no downloaded chips found in {root}. Run:\n'
                         f'  python run.py download --split {args.split} --modality s1')

    if args.order == 'water':
        rows.sort(key=lambda r: -r[1])
    elif args.order == 'least':
        rows.sort(key=lambda r: r[1])
    elif args.order == 'random':
        np.random.default_rng(0).shuffle(rows)

    rows = rows[:args.n]
    print(f'{len(rows)} chips from {args.split}  (ordered by: {args.order})\n')
    print(f'{"chip":<26}{"water %":>9}{"valid %":>9}')
    for c, w, v in rows:
        print(f'{c:<26}{w:>8.1f}%{v:>8.1f}%')

    # does this split have optical downloaded too?
    has_s2 = (root / 'S2Hand' / f'{rows[0][0]}_S2Hand.tif').exists()
    cols = ['Sentinel-1 VV', 'Sentinel-1 VH'] + (['Sentinel-2 RGB'] if has_s2 else []) + ['Label']
    if not has_s2:
        print('\n(no Sentinel-2 files found - showing SAR and label only)')

    n = len(rows)
    fig, ax = plt.subplots(n, len(cols), figsize=(2.6 * len(cols), 2.6 * n))
    if n == 1:
        ax = ax[None, :]

    for r, (chip, water, _) in enumerate(rows):
        with rasterio.open(root / 'S1Hand' / f'{chip}_S1Hand.tif') as src:
            s1 = src.read()
        with rasterio.open(root / 'LabelHand' / f'{chip}_LabelHand.tif') as src:
            lab = src.read(1)

        panels = [(stretch(s1[0]), dict(cmap='gray')),
                  (stretch(s1[1]), dict(cmap='gray'))]
        if has_s2:
            with rasterio.open(root / 'S2Hand' / f'{chip}_S2Hand.tif') as src:
                s2 = src.read()
            panels.append((np.dstack([stretch(s2[3]), stretch(s2[2]), stretch(s2[1])]), {}))
        panels.append((np.ma.masked_where(lab < 0, lab),
                       dict(cmap='bwr', vmin=0, vmax=1)))

        for c, (img, kw) in enumerate(panels):
            ax[r, c].imshow(img, **kw)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=10)
        ax[r, 0].set_ylabel(f'{chip}\n{water:.1f}% water', fontsize=7.5, rotation=0,
                            ha='right', va='center', labelpad=42)

    fig.suptitle(f'Sen1Floods11 - {args.split} split ({args.order} first)', fontsize=13)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches='tight')
    print(f'\nsaved {args.out}')
    plt.show()


if __name__ == '__main__':
    main()