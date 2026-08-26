"""Look at the STURM-Flood tiles that were pulled out of the archive.

    python scripts/explore_sturm.py

Two rows. The top row is Sentinel-1 tiles with their flood maps drawn over them,
the bottom row Sentinel-2 tiles with theirs. Laying them out this way makes the
point Section 3.2.3.3 argues in words: the two sensor subsets of STURM-Flood are
separate collections of different flood events, not two views of one scene, so
nothing here can be distilled from one to the other.

Scaling is left to a percentile stretch rather than fixed limits, because the
tiles are distributed already preprocessed and the units are not documented in
the same detail as Sen1Floods11.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def stretch(a, low=2, high=98):
    a = a.astype(np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def display(path):
    """Return something imshow can draw, whatever the band count turns out to be."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    data = np.nan_to_num(data)
    if data.shape[0] >= 3:
        return np.dstack([stretch(data[i]) for i in (2, 1, 0)]), {}
    return stretch(data[0]), dict(cmap='gray')


def load(folder, limit):
    img_dir, msk_dir = folder / 'img', folder / 'mask'
    if not img_dir.is_dir():
        raise SystemExit(f'{img_dir} not found. Fetch the tiles first:\n'
                         f'  python scripts/fetch_sturm.py --pairs '
                         f'{folder.name} --limit {limit}')
    names = sorted(p.name for p in img_dir.glob('*.tif'))[:limit]
    out = []
    for name in names:
        img, kw = display(img_dir / name)
        with rasterio.open(msk_dir / name) as src:
            msk = src.read(1)
        out.append((name, img, kw, msk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/sturm')
    ap.add_argument('--limit', type=int, default=4)
    ap.add_argument('--out', default='figures/sturm_overview.png')
    args = ap.parse_args()

    root = Path(args.root)
    rows = [('Sentinel-1', load(root / 's1', args.limit)),
            ('Sentinel-2', load(root / 's2', args.limit))]

    columns = max(len(r[1]) for r in rows)
    # tight_layout measures the axes but not the two-line titles above them, so
    # the lower row's titles land on top of the upper row's images. Letting
    # matplotlib do the constrained layout leaves room for both.
    fig, axes = plt.subplots(2, columns, figsize=(3.4 * columns, 8.2),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for r, (label, tiles) in enumerate(rows):
        for c in range(columns):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c >= len(tiles):
                ax.axis('off')
                continue
            name, img, kw, msk = tiles[c]
            ax.imshow(img, **kw)
            ax.imshow(np.ma.masked_where(msk <= 0, msk), cmap='autumn',
                      alpha=0.55, vmin=0, vmax=1)
            water = 100 * float((msk > 0).mean())
            event = name.split('_')[0]
            ax.set_title(f'{label}  {event}\nflooded {water:.0f} per cent', fontsize=9)

    fig.suptitle('STURM-Flood: Sentinel-1 tiles above, Sentinel-2 tiles below, '
                 'flood maps shown in yellow', fontsize=12)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f'saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
