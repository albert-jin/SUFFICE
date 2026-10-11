"""Download pinned public benchmarks outside the code repository; never models."""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def download(item, root):
    relative = Path('raw') / item['dataset'] / item['file']
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://huggingface.co/datasets/{}/resolve/{}/{}?download=true'.format(
        item['dataset'], item['revision'], urllib.parse.quote(item['file']))
    receipt = destination.with_name(destination.name + '.receipt.json')
    if destination.exists() and receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior.get('revision') == item['revision'] and prior.get('sha256') == sha256(destination):
            print('Verified ' + str(relative), flush=True)
            return prior
    partial = destination.with_name(destination.name + '.partial')
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'SUFFICE-dataset-preparation'})
            with urllib.request.urlopen(req, timeout=120) as response, partial.open('wb') as output:
                expected = response.headers.get('Content-Length')
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if expected and partial.stat().st_size != int(expected):
                raise IOError('Incomplete download: ' + str(relative))
            partial.replace(destination)
            record = {**item, 'path': relative.as_posix(), 'bytes': destination.stat().st_size,
                      'sha256': sha256(destination)}
            receipt.write_text(json.dumps(record, indent=2) + '\n')
            print('Downloaded ' + str(relative), flush=True)
            return record
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO.parent / 'SUFFICE-DATASETS')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        parser.error('Dataset output must be outside the repository.')
    sources = json.loads((REPO / 'configs/dataset_sources.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda item: download(item, output), sources))
    (output / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(records, indent=2) + '\n')
    print('Complete: {} pinned source files; {} bytes.'.format(len(records), sum(x['bytes'] for x in records)))

if __name__ == '__main__':
    main()
