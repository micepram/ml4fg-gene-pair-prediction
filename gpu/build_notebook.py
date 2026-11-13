"""Generate the two-model Colab launcher with a fixed three-hour total cap."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''
# Gene-pair prediction: three-hour training limit
Pramika Garg · ML4FG · Norman K562

This notebook compares **two matched full-cell models**: native GEARS and GEARS with
DepMap features plus set attention. Both predict all 19,264 genes, use 64 hidden
units, the official GO graph and upstream loss, and share split 101 and seed 17.
This is a preliminary comparison; repeated-split conclusions require later experiments.

**The combined allowance is at most three hours, not three hours per model.**
The timer starts with GPU checks/timing and includes fitting, validation, checkpoints
and scoring. Data downloading/preparation and CPU tests happen before that timer.
One common epoch count, at most 20, is chosen from measured GPU speed with a safety
margin. The supervisor saves early and force-stops training processes if needed.
A minute is deliberately left unused. An expired allowance starts no more GPU work.

Checkpoints and the original deadline are stored in `MyDrive/ML4FG_GPU/bounded_3h`.
Rerunning or reconnecting does **not** reset the allowance; pauses count too.
Keep that output folder fixed. No larger sweep is launched by this notebook.

Select **Runtime → Change runtime type → T4 GPU**, then run cells in order.
Google Drive may ask you to authorize access. Preparation needs about 15 GB free
VM disk. Only DepMap is prepared for these two models; GenePT is not downloaded.
GitHub pushes do not update an already-open notebook: save edits and reopen this
notebook to load changed cells. Training runs in fresh Python processes.
''')
code('''
# 1. Fetch code, install dependencies without replacing Colab's CUDA PyTorch.
from pathlib import Path
import os, subprocess, sys
REPO = Path('/content/ml4fg-gene-pair-prediction')
if not REPO.exists():
    result = subprocess.run(['git', 'clone', 'https://github.com/micepram/ml4fg-gene-pair-prediction.git', str(REPO)], capture_output=True, text=True)
else:
    result = subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], capture_output=True, text=True)
print(result.stdout, result.stderr)
result.check_returncode()
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from gpu.console import run_command
run_command([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'gpu/requirements-colab.txt'])
import torch
assert torch.cuda.is_available(), 'Select a CUDA GPU runtime before continuing.'
print('Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name())
''')
code('''
# 2. Persistent checkpoints and immutable deadline.
from google.colab import drive
drive.mount('/content/drive')
RUNS = Path('/content/drive/MyDrive/ML4FG_GPU/bounded_3h')
RUNS.mkdir(parents=True, exist_ok=True)
CACHE = REPO / 'gpu/cache'
BATCH_SIZE = 32
AMP = 'fp16'
MAX_TRAINING_HOURS = 3.0
assert 0 < MAX_TRAINING_HOURS <= 3
print('Outputs and saved allowance:', RUNS)
''')
code('''
# 3. Preparation and CPU checks. These do not start the training allowance.
# stdout and stderr are streamed explicitly so errors remain visible in Colab.
import json, shutil
if not (CACHE / 'READY.json').exists():
    assert shutil.disk_usage('/content').free > 15 * 2**30, 'Need about 15 GB free VM disk for first preparation.'
run_command([sys.executable, '-u', '-m', 'gpu.prepare', '--cache', str(CACHE), '--device', 'cuda', '--priors', 'depmap'])
ready = json.loads((CACHE / 'READY.json').read_text())
info = ready['coverage']['depmap']
print('DepMap targets:', info['covered_targets'], '/', info['targets'], 'missing:', info['missing_targets'])
print('Graph:', json.loads((CACHE / 'graphs.json').read_text()))
run_command([sys.executable, '-u', '-m', 'unittest', 'discover', '-s', 'gpu/tests', '-v'])
''')
md('''
## Run the bounded comparison
The next cell checks full-size CUDA forward/backward passes for both configurations
and measures optimizer-step time. It chooses the same epoch count for both models,
then trains each on every available training cell per epoch. GPU check updates are
thrown away; each actual model starts from the prespecified seed.

Only validation pairs select model weights. Test scores are produced after fitting.
A fit stopped early is labeled `time_limited`, with the number of completed epochs.
If measured speed cannot fit even one matched epoch safely, no full fit is started.
Runtime, memory and network failures cannot be ruled out, so checks fail before the
larger fit and safe checkpoints are retained. The process supervisor enforces the cap.
''')
code('''
# 4. START THE SHARED ALLOWANCE: at most three hours for both models together.
run_command([sys.executable, '-u', '-m', 'gpu.run_suite', '--cache', str(CACHE),
             '--output', str(RUNS), '--profile', 'bounded', '--batch-size', str(BATCH_SIZE),
             '--amp', AMP, '--max-hours', str(MAX_TRAINING_HOURS)])
''')
code('''
# 5. Inspect saved results without starting or extending training.
import pandas as pd
if (RUNS / 'budget.json').exists():
    print('Saved allowance:', json.loads((RUNS / 'budget.json').read_text()))
if (RUNS / 'summary.csv').exists():
    summary = pd.read_csv(RUNS / 'summary.csv')
    display(summary[summary.subset == 'top1000'])
else:
    print('No scored fits yet. Inspect the visible logs and saved checkpoints; no accuracy result is claimed.')
''')
code('''
# 6. Small results archive; checkpoints stay in Drive.
import zipfile
archive = Path('/content/ML4FG_GPU_results.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in RUNS.rglob('*'):
        if p.is_file() and p.suffix in ['.csv', '.json', '.txt']:
            z.write(p, p.relative_to(RUNS))
print('Results package:', archive)
# To download later:
# from google.colab import files
# files.download(str(archive))
''')
nb.cells = cells
nb.metadata = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                   language_info=dict(name='python'), accelerator='GPU',
                   colab=dict(name='Colab_Training.ipynb', provenance=[], gpuType='T4'))
nbf.validate(nb)
nbf.write(nb, ROOT / 'Colab_Training.ipynb')
print('Wrote bounded Colab_Training.ipynb')
