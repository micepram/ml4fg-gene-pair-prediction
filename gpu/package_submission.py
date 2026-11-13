"""Bundle source, report and compact scoring artifacts, without raw data/weights."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    paths = [ROOT / name for name in names if name and (ROOT / name).is_file()]
    results = ROOT / 'gpu/results/bounded_3h'
    paths += [results / 'evaluation_context.npz', *sorted(results.glob('*/predictions.npz'))]
    entries = []
    for path in paths:
        raw = path.read_bytes()
        entries.append(dict(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    manifest = dict(repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    files=entries, excluded='Raw AnnData, virtual environments, graph/cell caches and model checkpoints')
    output = ROOT / 'Interim_Report_Code.zip'
    temporary = output.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, entry in zip(paths, entries):
            archive.write(path, entry['path'])
        archive.writestr('SUBMISSION_MANIFEST.json', json.dumps(manifest, indent=2))
    temporary.replace(output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert not any('/data/raw/' in name or name.endswith('.pt') for name in archive.namelist())
    print(f'Packaged {len(entries)} files: {output.stat().st_size / 2**20:.1f} MB')


if __name__ == '__main__':
    main()
