"""Download the published weights into release_assets and verify every SHA-256."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEMO = {'frozen_L14_resize_seed42.pkl', 'model.safetensors', 'config.json', 'preprocessor_config.json'}


def valid(path, asset):
    if not path.is_file() or path.stat().st_size != asset['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == asset['sha256']


def download(file, dest, base_url):
    path = dest / file['name']
    if valid(path, file):
        return path
    with tempfile.NamedTemporaryFile(dir=dest, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        with urlopen(base_url + file['name'], timeout=120) as stream, tmp.open('wb') as handle:
            shutil.copyfileobj(stream, handle)
        if not valid(tmp, file):
            raise ValueError(f"Checksum mismatch: {file['name']}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='include all thirteen frozen models')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'release_models.json').read_text())
    dest = ROOT / 'release_assets'
    dest.mkdir(exist_ok=True)
    for asset in manifest['assets']:
        name = asset['name']
        if not args.all and name not in DEMO:
            continue
        path = dest / name
        if valid(path, asset):
            print('Verified', name)
            continue
        base_url = f"https://github.com/QueenofUSSR/TellFake/releases/download/{manifest['tag']}/"
        if 'parts' not in asset:
            download(asset, dest, base_url)
            continue
        chunks = [download(part, dest, base_url) for part in asset['parts']]
        with tempfile.NamedTemporaryFile(dir=dest, delete=False) as f:
            tmp = Path(f.name)
        try:
            with tmp.open('wb') as f:
                for chunk in chunks:
                    with chunk.open('rb') as stream:
                        shutil.copyfileobj(stream, f)
            if not valid(tmp, asset):
                raise ValueError(f'Checksum mismatch: {name}')
            os.replace(tmp, path)
            for chunk in chunks:
                chunk.unlink()
            print('Downloaded', name)
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
