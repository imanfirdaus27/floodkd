"""Pull a handful of STURM-Flood tiles without downloading the four gigabyte zip.

The whole dataset is one 3.71 GB file on Zenodo and the connection to it is slow,
which puts the full download somewhere around twelve hours. But the tiles inside
are 128 by 128 pixels — a few hundred kilobytes each — and all that is wanted
here is enough to look at.

So the zip is read where it sits. Its table of contents lives at the end of the
file and every entry records where its bytes begin, so individual tiles can be
pulled out by byte range over HTTP.

    python scripts/fetch_sturm.py --list
    python scripts/fetch_sturm.py --list --filter s1
    python scripts/fetch_sturm.py --get EMSR456 --limit 8 --out data/sturm

The reader itself is the one written for S1S2-Water; only the address differs.
"""
from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path

from fetch_s1s2 import HTTPRangeFile

URL = 'https://zenodo.org/records/12748983/files/Dataset.zip?download=1'

# How much to fetch in one request. The S1S2-Water files are gigabytes each, so
# a sixteen megabyte window keeps the number of requests down. STURM-Flood tiles
# are 132 kB, and the same window would download sixteen megabytes to obtain
# each one — slower by two orders of magnitude, and long enough for the
# connection to be dropped part way. Read a little at a time here.
CHUNK = 256 << 10


def archive():
    return zipfile.ZipFile(HTTPRangeFile(URL, chunk=CHUNK))


def contents():
    print('reading the table of contents (a few megabytes, not four gigabytes) ...',
          flush=True)
    with archive() as z:
        return z.infolist()


def show(infos, needle, depth, limit):
    if needle:
        infos = [i for i in infos if needle.lower() in i.filename.lower()]
    print(f'{len(infos)} entries match\n')

    shapes = Counter()
    for info in infos:
        parts = Path(info.filename).parts
        shapes['/'.join(parts[:depth])] += 1
    print('folder layout:')
    for name, count in sorted(shapes.items()):
        print(f'  {count:>7}  {name}')

    print('\nfirst entries:')
    for info in infos[:limit]:
        print(f'  {info.file_size / 1e3:8.1f} kB  {info.filename}')


def pairs(infos, sensor, limit, out):
    """Fetch N tiles of one sensor together with the mask of the same name.

    Taking the first N of the image folder and the first N of the mask folder
    separately would be easier and occasionally wrong: nothing guarantees the
    two folders are stored in the same order inside the zip. Matching by name
    removes the doubt.
    """
    band = {'s1': 'Sentinel1', 's2': 'Sentinel2'}[sensor.lower()]
    inner = band.replace('entinel', '')            # Sentinel1 -> S1
    images = {Path(i.filename).name: i for i in infos
              if f'{band}/{inner}/' in i.filename and not i.is_dir()}
    masks = {Path(i.filename).name: i for i in infos
             if f'{band}/Floodmaps/' in i.filename and not i.is_dir()}

    shared = sorted(set(images) & set(masks))
    if not shared:
        raise SystemExit(f'no tile of {band} has a matching flood map')
    print(f'{len(shared)} {band} tiles have a matching flood map')

    # spread the choice across the archive rather than taking one flood event
    step = max(len(shared) // limit, 1)
    chosen = shared[::step][:limit]

    img_dir = out / sensor.lower() / 'img'
    msk_dir = out / sensor.lower() / 'mask'
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    with archive() as z:
        for name in chosen:
            for info, folder in ((images[name], img_dir), (masks[name], msk_dir)):
                dst = folder / name
                if dst.exists() and dst.stat().st_size == info.file_size:
                    continue
                print(f'  get   {folder.name}/{name}  '
                      f'({info.file_size / 1e3:.0f} kB)', flush=True)
                with z.open(info) as src, open(dst, 'wb') as f:
                    f.write(src.read())
    print(f'\n{len(chosen)} tiles and {len(chosen)} masks under {out / sensor.lower()}')


def get(infos, needle, limit, out):
    wanted = [i for i in infos
              if needle.lower() in i.filename.lower() and not i.is_dir()][:limit]
    if not wanted:
        raise SystemExit(f'nothing matched {needle!r}; try --list first')

    out.mkdir(parents=True, exist_ok=True)
    with archive() as z:
        for info in wanted:
            dst = out / Path(info.filename).name
            if dst.exists() and dst.stat().st_size == info.file_size:
                print(f'  have  {dst.name}')
                continue
            print(f'  get   {dst.name}  ({info.file_size / 1e3:.0f} kB)', flush=True)
            with z.open(info) as src, open(dst, 'wb') as f:
                f.write(src.read())
    print(f'\n{len(wanted)} files in {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--filter', default='', help='only entries containing this text')
    ap.add_argument('--depth', type=int, default=2, help='folder levels to summarise')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--get', help='download entries containing this text')
    ap.add_argument('--pairs', choices=['s1', 's2'],
                    help='download tiles of one sensor with their flood maps')
    ap.add_argument('--out', default='data/sturm')
    args = ap.parse_args()

    infos = contents()
    if args.pairs:
        pairs(infos, args.pairs, args.limit, Path(args.out))
    elif args.get:
        get(infos, args.get, args.limit, Path(args.out))
    else:
        show(infos, args.filter, args.depth, args.limit)


if __name__ == '__main__':
    main()
