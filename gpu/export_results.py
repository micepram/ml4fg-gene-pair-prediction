"""Export completed results and scoring context, without model checkpoints."""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from .common import atomic_json, sha256


def export(cache, runs, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [(p, str(p.relative_to(runs))) for p in sorted(runs.rglob('*'))
             if p.is_file() and (p.suffix in {'.csv', '.json', '.txt'} or p.name == 'predictions.npz')]
    if not list(runs.glob('*/result.json')):
        raise ValueError('No scored fits to export; this command does not launch training')
    profiles = np.load(cache / 'profiles.npz', allow_pickle=False)
    with tempfile.TemporaryDirectory() as temporary:
        context = Path(temporary) / 'evaluation_context.npz'
        np.savez_compressed(context, means=profiles['means'], conditions=profiles['conditions'],
                            genes=profiles['genes'], top1000=profiles['top1000'], counts=profiles['counts'],
                            top20=np.load(cache / 'top20.npy', allow_pickle=False))
        paths.append((context, context.name))
        paths += [(cache / name, 'data_provenance/' + name)
                  for name in ['source_manifest.json', 'graphs.json', 'depmap.json', 'splits.json']]
        entries = [dict(path=name, bytes=path.stat().st_size, sha256=sha256(path)) for path, name in paths]
        manifest = Path(temporary) / 'artifact_manifest.json'
        atomic_json(dict(files=entries, checkpoints='Latest and best weights remain in Google Drive',
                         purpose='Independent numerical verification of completed, fixed-budget fits'), manifest)
        temporary_zip = output.with_suffix('.zip.part')
        with zipfile.ZipFile(temporary_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path, name in paths:
                archive.write(path, name)
            archive.write(manifest, manifest.name)
        temporary_zip.replace(output)
    print(f'Exported {len(entries)} verified entries to {output} ({output.stat().st_size/2**20:.1f} MB)', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    export(args.cache, args.runs, args.output)
