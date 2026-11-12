from pathlib import Path
import hashlib
import json
import os
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / 'interim'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def load_checkpoint(path, device='cpu'):
    # These are our own checkpoints, including Python/NumPy RNG states.
    return torch.load(path, map_location=device, weights_only=False)


def provenance():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    return dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')),
                source_hashes={p.name: sha256(p) for p in sorted((ROOT / 'gpu').glob('*.py'))},
                upstream_commit=(INTERIM / 'vendor/GEARS/COMMIT').read_text().strip())
