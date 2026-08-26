"""Look at one S1S2-Water scene.

    python scripts/explore_s1s2.py --sample 1

Produces one figure with two rows. The top row is the whole 10980 by 10980 scene
shrunk to fit: radar VV, radar VH, optical true colour, and the water mask. The
bottom row is a 512 by 512 crop taken from wherever the mask says there is the
most water, at full resolution, so the detail is actually visible.

Nothing is loaded at full size except the crop. The overview is read decimated,
which is what makes a two gigabyte scene openable on a laptop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

# S1S2-Water stores radar in decibels multiplied by 100, and optical as
# top-of-atmosphere reflectance multiplied by 10000.
S1_SCALE, S1_MIN, S1_MAX = 100.0, -50.0, 1.0
S2_SCALE = 10000.0
# The six stored bands are Blue, Green, Red, NIR, SWIR1, SWIR2. rasterio counts
# bands from one, so true colour is band 3, then 2, then 1.
TRUE_COLOUR = (3, 2, 1)
CROP = 512


def stretch(a, low=2, high=98):
    """Percentile stretch, so a dark radar scene is actually visible."""
    a = a.astype(np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def overview(path, bands, size):
    """Read a decimated version of the whole raster."""
    with rasterio.open(path) as src:
        return src.read(bands, out_shape=(len(bands), size, size)).astype(np.float32)


def busiest_window(mask_small, factor, crop, valid_small=None):
    """A crop-sized window that shows a shoreline, not a blank patch of sea.

    Picking the window with the most water is the obvious thing to do and it is
    wrong: it lands in the middle of open water, where every panel is a flat
    colour and half the tile can be outside the satellite swath. What is wanted
    is a window that is entirely inside valid data and roughly half water, so
    the radar signature of the boundary is actually visible.
    """
    side = max(int(round(crop / factor)), 4)
    water = np.pad(mask_small.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    if valid_small is None:
        valid_small = np.ones_like(mask_small, dtype=bool)
    good = np.pad(valid_small.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def total(table, r, c):
        return (table[r + side, c + side] - table[r, c + side]
                - table[r + side, c] + table[r, c])

    cells = side * side
    best_score, best_rc, best_water = -1.0, (0, 0), 0
    step = max(side // 4, 1)
    for r in range(0, mask_small.shape[0] - side, step):
        for c in range(0, mask_small.shape[1] - side, step):
            if total(good, r, c) < 0.99 * cells:
                continue
            fraction = total(water, r, c) / cells
            score = 1.0 - abs(fraction - 0.5) * 2.0
            if score > best_score:
                best_score, best_rc, best_water = score, (r, c), fraction
    return int(best_rc[0] * factor), int(best_rc[1] * factor), best_water


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default='1')
    ap.add_argument('--root', default='data/s1s2water')
    ap.add_argument('--size', type=int, default=1100, help='overview side in pixels')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    folder = Path(args.root) / args.sample
    s1_path = folder / f'sentinel12_s1_{args.sample}_img.tif'
    s2_path = folder / f'sentinel12_s2_{args.sample}_img.tif'
    msk_path = folder / f'sentinel12_s1_{args.sample}_msk.tif'
    for p in (s1_path, s2_path, msk_path):
        if not p.exists():
            raise SystemExit(f'missing {p}\nFetch it first:  python scripts/fetch_s1s2.py '
                             f'--sample {args.sample} --out {args.root}')

    meta_path = folder / f'sentinel12_{args.sample}_meta.json'
    place = ''
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        props = meta.get('properties', {})
        place = str(props.get('datetime', ''))[:10]

    with rasterio.open(s1_path) as src:
        height, width = src.height, src.width
    print(f'scene {args.sample}: {width} x {height} px')

    size = args.size
    factor = width / size

    s1 = overview(s1_path, [1, 2], size) / S1_SCALE
    s1 = (np.clip(s1, S1_MIN, S1_MAX) - S1_MIN) / (S1_MAX - S1_MIN)
    s2 = overview(s2_path, list(TRUE_COLOUR), size)
    s2 = np.clip(s2 / S2_SCALE, 0, 1)
    msk = overview(msk_path, [1], size)[0]

    water = 100 * float((msk > 0).mean())
    print(f'water in this scene: {water:.1f} per cent')

    valid_path = folder / f'sentinel12_s1_{args.sample}_valid.tif'
    valid_small = None
    if valid_path.exists():
        valid_small = overview(valid_path, [1], size)[0] > 0

    row, col, crop_water = busiest_window(msk > 0, factor, CROP, valid_small)
    row = min(max(row, 0), height - CROP)
    col = min(max(col, 0), width - CROP)
    print(f'crop taken at row {row}, column {col}, '
          f'about {100 * crop_water:.0f} per cent water')

    win = Window(col, row, CROP, CROP)
    with rasterio.open(s1_path) as src:
        s1c = src.read([1, 2], window=win).astype(np.float32) / S1_SCALE
    s1c = (np.clip(s1c, S1_MIN, S1_MAX) - S1_MIN) / (S1_MAX - S1_MIN)
    with rasterio.open(s2_path) as src:
        s2c = src.read(list(TRUE_COLOUR), window=win).astype(np.float32)
    s2c = np.clip(s2c / S2_SCALE, 0, 1)
    with rasterio.open(msk_path) as src:
        mskc = src.read(1, window=win)

    panels = [
        (stretch(s1[0]), 'Sentinel-1 VV', dict(cmap='gray')),
        (stretch(s1[1]), 'Sentinel-1 VH', dict(cmap='gray')),
        (np.dstack([stretch(s2[i]) for i in range(3)]), 'Sentinel-2 true colour', {}),
        (np.ma.masked_where(msk <= 0, msk), 'Water mask', dict(cmap='bwr', vmin=0, vmax=1)),
        (stretch(s1c[0]), 'VV, 512 px crop', dict(cmap='gray')),
        (stretch(s1c[1]), 'VH, 512 px crop', dict(cmap='gray')),
        (np.dstack([stretch(s2c[i]) for i in range(3)]), 'Optical, 512 px crop', {}),
        (np.ma.masked_where(mskc <= 0, mskc), 'Mask, 512 px crop',
         dict(cmap='bwr', vmin=0, vmax=1)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.4))
    for ax, (img, title, kw) in zip(axes.ravel(), panels):
        ax.imshow(img, **kw)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'S1S2-Water scene {args.sample}'
                 + (f'  ({place})' if place else '')
                 + f'   water {water:.1f} per cent', fontsize=13)
    fig.tight_layout()

    out = Path(args.out or f'figures/s1s2water_{args.sample}.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
