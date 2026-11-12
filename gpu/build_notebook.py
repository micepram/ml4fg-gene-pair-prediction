"""Generate the clean Colab launcher (no credentials or claimed GPU results)."""
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
# Full cell-level training on Colab
Pramika Garg · ML4FG · Norman K562 unseen gene combinations

This notebook **downloads data and trains models on the connected CUDA GPU**.
It is separate from `interim/Interim_Analysis.ipynb`, which reads completed local pilots.

Use **Runtime → Change runtime type → T4 GPU** (or another available CUDA GPU),
then run these cells in order. Google Drive authorization in cell 2 requires your click.
Checkpoints and results stay in your Drive; raw data and the ~6.5 GB cell cache stay on the VM.
Preparation requires ~15 GB free VM disk and can take several minutes.

The default extended plan has **70 full fits across five pair partitions and two seeds**:
10 native GEARS reference fits at 64 dimensions/20 epochs, plus the six matched
configurations at 128 dimensions/50 epochs. This needs multiple Colab sessions.
Each session stops after the configured training budget and saves progress every five minutes.
Rerun the notebook to continue; completed fits are checked and skipped. GPU access and runtime
duration are variable, so checkpointing cannot guarantee uninterrupted execution.

GitHub pushes do **not** update an already-open notebook. Save any edits, reopen this
notebook for changed cells, then rerun setup to pull scripts. Do not pull during a fit.
The training runs in fresh subprocesses, avoiding stale imported model code.
''')
code('''
# 1. Clone/update the code. This keeps Colab's existing CUDA PyTorch installation.
from pathlib import Path
import os, subprocess, sys, time
SESSION_START = time.monotonic()
REPO = Path('/content/ml4fg-gene-pair-prediction')
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/micepram/ml4fg-gene-pair-prediction.git', str(REPO)], check=True)
else:
    # ff-only preserves history; Git stops rather than overwriting conflicting local edits.
    subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
os.chdir(REPO)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'gpu/requirements-colab.txt'], check=True)
import torch
assert torch.cuda.is_available(), 'Connect a GPU runtime before continuing.'
print('Code:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(), 'VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
''')
code('''
# 2. Persistent checkpoints. Accept the Google Drive authorization yourself.
from google.colab import drive
drive.mount('/content/drive')
RUNS = Path('/content/drive/MyDrive/ML4FG_GPU/runs')
RUNS.mkdir(parents=True, exist_ok=True)
CACHE = REPO / 'gpu/cache'
print('Persistent outputs:', RUNS)
''')
md('''
## Experiment settings
`extended` uses the larger models and longer training. `upstream` runs the six
configurations at 64 dimensions/20 epochs with upstream learning-rate decay (60 fits).
Keep the preset, physical batch size and precision policy fixed when resuming an existing
suite. Changing them requires a **new output folder**; checkpoint compatibility is enforced.
Select hyperparameters using validation pairs only; do not choose presets using test results.
Eight hours is a training budget, not a promised Colab session length. Reduce it if this VM
has already been running for hours. No paid tier or compute purchase is required by this code.
''')
code('''
# 3. Settings shared across all matched configurations.
PROFILE = 'extended'
BATCH_SIZE = 32
AMP = 'fp16'  # T4-compatible mixed precision; 'off' uses FP32.
SESSION_HOURS = 8.0
SPLITS = [101, 202, 303, 404, 505]
SEEDS = [17, 29]
assert PROFILE in ['upstream', 'extended']
''')
code('''
# 4. Public downloads, cell targets, training-only coexpression, official GO graph,
#    frozen GenePT and labeled DepMap22Q4 features with K562/derivatives excluded.
import shutil
assert shutil.disk_usage('/content').free > 15 * 2**30, 'Need ~15 GB free VM disk for first preparation.'
subprocess.run([sys.executable, '-u', '-m', 'gpu.prepare', '--cache', str(CACHE), '--device', 'cuda'], check=True)
import json
ready = json.loads((CACHE / 'READY.json').read_text())
for name, info in ready['coverage'].items():
    print(name, 'target coverage:', info['covered_targets'], '/', info['targets'], 'missing:', info['missing_targets'])
print(json.loads((CACHE / 'graphs.json').read_text()))
''')
code('''
# 5. Meaningful CPU fixture tests before consuming GPU time on full fits.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'gpu/tests', '-v'], check=True)
subprocess.run([sys.executable, '-u', '-m', 'gpu.smoke', '--cache', str(CACHE), '--batch-size', str(BATCH_SIZE)], check=True)
''')
md('''
## Train and resume
This cell first runs the native GEARS reference, then attention + DepMap (the prespecified
primary extension), then the other matched configurations. All cells from a pair stay in
one partition. Only training cells enter the objective; validation pairs select the best
epoch. Test scores are produced after each fit completes, and must not guide later choices.

The original upstream forward is retained, including its graph batching behavior.
Inference uses a fixed batch size of one. Because GEARS adds input expression only at its
output, evaluating the control mean gives the exact average over individual control inputs
at that inference batch size. Training still uses individual target cells and fixed unpaired
control-cell inputs. The preparation stage does not train on condition means.

The processed benchmark release, validation metric, mixed precision and mean inference
are documented protocol differences; this is not a claim of reproducing every paper number.
''')
code('''
# 6. Full fits. Interrupting requests a checkpoint after the current optimizer step.
remaining_hours = min(SESSION_HOURS, max(0, 9.0 - (time.monotonic() - SESSION_START) / 3600))
assert remaining_hours > 0, 'Start a fresh permitted GPU session before resuming.'
command = [sys.executable, '-u', '-m', 'gpu.run_suite', '--cache', str(CACHE),
           '--output', str(RUNS), '--profile', PROFILE, '--batch-size', str(BATCH_SIZE),
           '--amp', AMP, '--max-hours', str(remaining_hours),
           '--splits', *map(str, SPLITS), '--seeds', *map(str, SEEDS)]
subprocess.run(command, check=True)
''')
code('''
# 7. Summarize completed fits. Missing splits/seeds are explicitly provisional.
subprocess.run([sys.executable, '-m', 'gpu.summarize', '--runs', str(RUNS)], check=True)
import pandas as pd
if (RUNS / 'summary.csv').exists():
    summary = pd.read_csv(RUNS / 'summary.csv')
    display(summary[summary.subset == 'top1000'])
else:
    print('No completed full-cell fits yet; rerun training to continue from saved checkpoints.')
''')
code('''
# 8. Optional small results download; model checkpoints stay in your Drive.
import zipfile
from google.colab import files
archive = Path('/content/ML4FG_GPU_results.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in RUNS.rglob('*'):
        if p.is_file() and p.suffix in ['.csv', '.json', '.txt']:
            z.write(p, p.relative_to(RUNS))
print('Results package:', archive)
# Uncomment when you want the download:
# files.download(str(archive))
''')
nb.cells = cells
nb.metadata = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                   language_info=dict(name='python'), accelerator='GPU',
                   colab=dict(name='Colab_Training.ipynb', provenance=[], gpuType='T4'))
nbf.validate(nb)
nbf.write(nb, ROOT / 'Colab_Training.ipynb')
print('Wrote Colab_Training.ipynb')
