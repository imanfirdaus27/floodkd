"""Plot training curves from runs/*/log.csv.

Lives in scripts/. Run it FROM THE PROJECT ROOT (the folder with run.py):

    python scripts/plot.py                          every run in runs/
    python scripts/plot.py --runs runs/base_s1      just one
    python scripts/plot.py --out slide.png          choose the filename

Produces a two-panel figure: validation IoU on the left, loss on the right.
The best epoch of each run is marked with a dot and printed in the legend,
which is what you want on a slide.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))   # so `src` is importable


COLOURS = ['#2E5F8A', '#C55A11', '#4C7A34', '#6B4E8F', '#A63A3A']


def read_log(path: Path) -> dict:
    rows = list(csv.DictReader(open(path)))
    out = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            v = r[key]
            try:
                vals.append(float(v) if v not in ('', 'None') else None)
            except ValueError:
                vals.append(None)
        out[key] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--out', default='figures/training_curve.png')
    args = ap.parse_args()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        run_dirs = sorted(p.parent for p in Path('runs').glob('*/log.csv'))

    run_dirs = [d for d in run_dirs if (d / 'log.csv').exists()]
    if not run_dirs:
        raise SystemExit('no runs/*/log.csv found. Train something first.')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    print(f'{"run":<22}{"epochs":>8}{"best IoU":>11}{"at epoch":>10}')
    for i, d in enumerate(run_dirs):
        log = read_log(d / 'log.csv')
        colour = COLOURS[i % len(COLOURS)]
        ep = log['epoch']
        iou = log['val_iou_water']

        best = max((v, e) for v, e in zip(iou, ep) if v is not None)
        print(f'{d.name:<22}{len(ep):>8}{best[0]:>11.4f}{int(best[1]):>10}')

        ax1.plot(ep, iou, color=colour, lw=1.8,
                 label=f'{d.name}  (best {best[0]:.4f})')
        ax1.scatter([best[1]], [best[0]], color=colour, zorder=5, s=45)
        ax1.annotate(f'{best[0]:.3f}', (best[1], best[0]),
                     textcoords='offset points', xytext=(6, 6),
                     fontsize=9, color=colour)

        ax2.plot(ep, log['train_loss'], color=colour, lw=1.8, label=f'{d.name} train')
        if any(v is not None for v in log.get('val_loss', [])):
            ax2.plot(ep, log['val_loss'], color=colour, lw=1.4, ls='--',
                     label=f'{d.name} val')

    ax1.set_xlabel('epoch'); ax1.set_ylabel('validation IoU (water)')
    ax1.set_title('Validation IoU')
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)

    ax2.set_xlabel('epoch'); ax2.set_ylabel('loss')
    ax2.set_title('Loss')
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'\nsaved {args.out}')
    plt.show()


if __name__ == '__main__':
    main()