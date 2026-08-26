"""Pull single S1S2-Water scenes out of the Zenodo zips without downloading them.

The dataset is 165 GB in six zip files and the smallest part is 19.8 GB, so the
obvious route — download a part, unzip, look at one scene — costs twenty
gigabytes to see a picture.

It does not have to. A zip file keeps its table of contents at the end, and
every member records where its bytes start. Zenodo serves HTTP range requests,
so the table can be read on its own and then individual members pulled out by
byte range. One scene is roughly one to two gigabytes instead of twenty.

    python scripts/fetch_s1s2.py --list
    python scripts/fetch_s1s2.py --sample 1 --out data_s1s2
    python scripts/fetch_s1s2.py --sample 1 --out data_s1s2 --layers s1_img,s1_msk

Nothing here is specific to Zenodo. Any server that answers range requests works.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RECORD = 'https://zenodo.org/records/11278238/files'
PARTS = [f'part{i}.zip' for i in range(1, 7)]
CACHE = Path('.s1s2_index.json')

# Zenodo drops connections that arrive with the default Python user agent, and
# it drops ordinary ones too when it is busy. Both have to be handled: send a
# browser-like header, and treat a reset as something to retry rather than as
# the end of the run.
HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                   '(KHTML, like Gecko) Chrome/126.0 Safari/537.36'),
    'Accept': '*/*',
    'Connection': 'close',
}
ATTEMPTS = 6


# Zenodo refuses Python's TLS handshake from some networks — the connection is
# reset before any HTTP is spoken — while curl on the same machine connects
# happily. Where curl exists it is therefore the transport, and urllib is the
# fallback rather than the other way round.
CURL = shutil.which('curl') or shutil.which('curl.exe')


def curl_range(url, start, end):
    """Fetch bytes start..end inclusive, and confirm the server obeyed."""
    proc = subprocess.run(
        [CURL, '-sS', '-L', '--fail', '--retry', '5', '--retry-delay', '2',
         '--retry-connrefused', '-i', '--range', f'{start}-{end}', url],
        capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise OSError(proc.stderr.decode('utf8', 'replace')[:200])

    head, _, body = proc.stdout.partition(b'\r\n\r\n')
    while head.startswith(b'HTTP/') and b'\r\n\r\n' in body + b'\r\n\r\n' and (
            b' 30' in head.split(b'\r\n')[0] or b' 100 ' in head.split(b'\r\n')[0]):
        head, _, body = body.partition(b'\r\n\r\n')     # step past redirects
    status = head.split(b'\r\n')[0]
    if b'206' not in status:
        raise OSError(f'server ignored the range request: {status[:60]!r}')
    return head, body


def curl_size(url):
    head, _ = curl_range(url, 0, 0)
    match = re.search(rb'(?im)^content-range:\s*bytes\s+\d+-\d+/(\d+)', head)
    if not match:
        raise OSError('no Content-Range header, cannot tell the file size')
    return int(match.group(1))


def request(url, headers=None, method='GET'):
    """One HTTP request, retried with a growing pause if the server hangs up."""
    merged = dict(HEADERS)
    merged.update(headers or {})
    last = None
    for attempt in range(ATTEMPTS):
        try:
            req = urllib.request.Request(url, headers=merged, method=method)
            return urllib.request.urlopen(req, timeout=120)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            last = exc
            wait = 2 ** attempt
            print(f'    connection failed ({exc.__class__.__name__}), '
                  f'retrying in {wait}s', flush=True)
            time.sleep(wait)
    raise SystemExit(f'gave up after {ATTEMPTS} attempts: {last}')

# the four layers actually needed to train and to look at a scene
DEFAULT_LAYERS = ('s1_img', 's1_msk', 's1_valid', 's2_img', 's2_msk', 's2_valid')


class HTTPRangeFile(io.RawIOBase):
    """A read-only seekable file backed by HTTP range requests.

    zipfile only ever needs seek, tell and read, so that is all this provides.
    Everything it reads is a slice of a remote file that is never downloaded
    whole.
    """

    # Each range request opens a fresh connection, so the window has to be big.
    # At one megabyte a 1.7 GB optical image needs 1,700 handshakes and crawls;
    # at sixteen it needs about a hundred.
    def __init__(self, url: str, chunk: int = 16 << 20):
        self.url = url
        self.pos = 0
        self.chunk = chunk
        self._size = None
        self._buf = b''
        self._buf_start = -1

    # ---------------------------------------------------------------- basics
    def _head(self) -> int:
        if CURL:
            return curl_size(self.url)
        with request(self.url, method='HEAD') as r:
            return int(r.headers['Content-Length'])

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = self._head()
        return self._size

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    # ----------------------------------------------------------------- reads
    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        if end < start:
            return b''
        if CURL:
            for attempt in range(ATTEMPTS):
                try:
                    return curl_range(self.url, start, end)[1]
                except (OSError, subprocess.TimeoutExpired) as exc:
                    wait = 2 ** attempt
                    print(f'    curl failed ({str(exc)[:60]}), retrying in {wait}s',
                          flush=True)
                    time.sleep(wait)
            raise SystemExit('curl could not read the range after several attempts')

        for attempt in range(ATTEMPTS):
            with request(self.url, headers={'Range': f'bytes={start}-{end}'}) as r:
                if r.status != 206:
                    raise OSError('server ignored the range request; '
                                  'it would send the whole file')
                try:
                    return r.read()
                except (ConnectionError, TimeoutError, OSError) as exc:
                    # the handshake succeeded but the body was cut short
                    wait = 2 ** attempt
                    print(f'    transfer cut short ({exc.__class__.__name__}), '
                          f'retrying in {wait}s', flush=True)
                    time.sleep(wait)
        raise SystemExit('the server kept closing the connection mid-transfer')

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b''
        # small reads come out of a one-megabyte window, so parsing the zip
        # directory does not turn into a thousand requests
        if n <= self.chunk:
            if not (self._buf_start <= self.pos
                    and self.pos + n <= self._buf_start + len(self._buf)):
                self._buf_start = self.pos
                self._buf = self._fetch(self.pos, self.chunk)
            off = self.pos - self._buf_start
            data = self._buf[off:off + n]
        else:
            data = self._fetch(self.pos, n)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def index(refresh: bool = False) -> dict:
    """Map every sample id to the part that holds it. Cached after the first run."""
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text())

    found = {}
    for part in PARTS:
        url = f'{RECORD}/{part}?download=1'
        print(f'reading the table of contents of {part} ...', flush=True)
        with zipfile.ZipFile(HTTPRangeFile(url)) as z:
            for name in z.namelist():
                bits = Path(name).parts
                if len(bits) >= 2 and bits[-2].isdigit():
                    found.setdefault(bits[-2], part)
    CACHE.write_text(json.dumps(found, indent=1, sort_keys=True))
    print(f'{len(found)} samples indexed -> {CACHE}')
    return found


def fetch(sample: str, out: Path, layers) -> list[Path]:
    where = index()
    if sample not in where:
        raise SystemExit(f'sample {sample} not found. Known: '
                         f'{", ".join(sorted(where, key=int))}')
    part = where[sample]
    url = f'{RECORD}/{part}?download=1'
    out.mkdir(parents=True, exist_ok=True)

    # every file is named sentinel12_SENSOR_ID_LAYER.tif, so the wanted set can
    # be written out directly rather than guessed at with string matching
    names = {f'sentinel12_{sensor}_{sample}_{layer}.tif'
             for sensor, layer in (lay.split('_', 1) for lay in layers)}
    names.add(f'sentinel12_{sample}_meta.json')

    written = []
    with zipfile.ZipFile(HTTPRangeFile(url)) as z:
        wanted = [info for info in z.infolist()
                  if Path(info.filename).name in names]

        if not wanted:
            raise SystemExit(f'no matching layers for sample {sample}')

        total = sum(i.file_size for i in wanted)
        print(f'sample {sample} is in {part}; fetching {len(wanted)} files '
              f'({total / 1e9:.2f} GB uncompressed)')

        for info in wanted:
            dst = out / sample / Path(info.filename).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size == info.file_size:
                print(f'  have  {dst.name}')
                written.append(dst)
                continue
            # the optical image alone is about 1.7 GB, so silence here looks
            # exactly like a hang. Report progress on the same line.
            mb = info.file_size / 1e6
            print(f'  get   {dst.name}  ({mb:.0f} MB)', flush=True)
            done = 0
            with z.open(info) as src, open(dst, 'wb') as f:
                while True:
                    block = src.read(1 << 22)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    if mb > 20:
                        pct = 100 * done / max(info.file_size, 1)
                        print(f'\r        {done / 1e6:7.0f} / {mb:.0f} MB  '
                              f'{pct:5.1f}%', end='', flush=True)
            if mb > 20:
                print()
            written.append(dst)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--list', action='store_true', help='index the parts and stop')
    p.add_argument('--refresh', action='store_true', help='rebuild the index')
    p.add_argument('--sample', help='sample id, e.g. 1')
    p.add_argument('--out', default='data_s1s2')
    p.add_argument('--layers', default=','.join(DEFAULT_LAYERS))
    args = p.parse_args()

    if args.list or not args.sample:
        where = index(refresh=args.refresh)
        by_part = {}
        for sample, part in where.items():
            by_part.setdefault(part, []).append(sample)
        for part in PARTS:
            ids = sorted(by_part.get(part, []), key=int)
            print(f'{part}: {len(ids)} samples  {", ".join(ids)}')
        return

    files = fetch(args.sample, Path(args.out), args.layers.split(','))
    print(f'\n{len(files)} files in {Path(args.out) / args.sample}')


if __name__ == '__main__':
    main()
