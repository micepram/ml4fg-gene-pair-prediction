"""Build and execute a lightweight notebook over the actual saved experiments."""
from pathlib import Path
import json
import sys
import tempfile
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.cells=[
md('''# Predicting gene-pair responses beyond additivity
**Pramika Garg · pg2899 · Interim experiments**

This executed notebook reads the saved results produced by the accompanying scripts.
It does not download data or retrain models when run. See `README.md` for complete rerun commands.
The models are explicitly condition-mean GEARS architecture pilots, not a reproduction of the published single-cell training procedure.'''),
code('''from pathlib import Path
import json, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
ROOT = Path.cwd()
if not (ROOT / 'results').exists():
    ROOT = ROOT / 'interim'
assert (ROOT / 'results').exists()
summary = json.loads((ROOT / 'data/processed/data_summary.json').read_text())
pd.Series({k: summary[k] for k in ['original_cells','retained_cells','genes','original_conditions','retained_conditions','singles','pairs','control_cells','excluded_cells','zero_fraction']}, name='Measured value')'''),
md('''## Data and preprocessing
The source AnnData already contains continuous, nonnegative expression with `log1p` metadata.
No second normalization is applied. Conditions containing IER5L are excluded because that target
is missing from the supplied GO dictionary. PCA is descriptive and is never fed into the model.'''),
code('''conditions = pd.read_csv(ROOT / 'data/processed/conditions.csv')
display(conditions.groupby('type').n_cells.agg(['count','sum','min','median','max']))
print('Excluded conditions:', summary['excluded_conditions'])
print('Nonfinite stored values:', summary['nonfinite_values'])
print('PCA explained variance:', np.round(summary['pca_variance_ratio'][:2], 4))
display(Image(filename=str(ROOT / 'figures/02_eda.png')))'''),
md('''## Pair partitions and evaluation
All single-gene perturbations are observed during training. Each pair is assigned to exactly
one train/validation/test partition within a split. Repeated test appearances are not independent
samples. Positive paired improvement means `MSE_additive - MSE_model > 0`.'''),
code('''splits = json.loads((ROOT / 'data/processed/splits.json').read_text())
rows = []
for s in splits:
    tr, va, te = map(set, [s['train'], s['validation'], s['test']])
    assert not (tr & va or tr & te or va & te)
    rows.append({'partition_seed':s['seed'], 'train_pairs':sum('+' in c for c in tr), 'validation_pairs':len(va), 'test_pairs':len(te)})
display(pd.DataFrame(rows))
print('Unique tested pairs:', len({c for s in splits for c in s['test']}))'''),
md('''## Simple baselines across five partitions
The training mean weights perturbation conditions equally and excludes controls.
No-change prediction has zero control-relative shift, so its Pearson correlation is undefined.
The primary result uses MSE, avoiding that undefined correlation.'''),
code('''baseline = pd.read_csv(ROOT / 'results/baseline_summary.csv')
display(baseline)
display(pd.read_csv(ROOT / 'results/baseline_split_metrics.csv'))'''),
md('''## Model pilots and embedding coverage
Both models use partition 101 and initialization seed 17, with validation-based checkpoint selection.
They use condition means, ordinary MSE, 32 hidden units, all 19,264 genes and a 100-target GO graph.
These departures from the published cell-level protocol prevent treating this as its reproduction.
The GenePT pilot adds fixed NCBI-description features through a learned projection and scale.'''),
code('''runs = [json.loads((ROOT / f'results/{slug}_run.json').read_text()) for slug in ['gears','gears_genept']]
display(pd.DataFrame(runs)[['method','split','seed','epochs_completed','best_epoch','validation_mse','test_mse','hidden','readout_genes','parameters','wall_seconds']])
display(pd.Series(json.loads((ROOT / 'results/embedding_coverage.json').read_text())))
display(pd.read_csv(ROOT / 'results/common_split_comparison.csv'))'''),
code('''fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for slug, label in [('gears','GEARS pilot'), ('gears_genept','GEARS + GenePT pilot')]:
    h = pd.read_csv(ROOT / f'results/{slug}_training_history.csv')
    axes[0].plot(h.epoch, h.training_mse, label=label)
    axes[1].plot(h.epoch, h.validation_mse, label=label)
for ax, title in zip(axes, ['Training: all-gene MSE', 'Validation: primary 1,000-gene MSE']):
    ax.set(xlabel='Epoch', ylabel='MSE', title=title)
    ax.legend()
fig.tight_layout()
plt.show()'''),
code("display(Image(filename=str(ROOT / 'figures/03_performance.png')))"),
md('''## Interaction repeatability
The two residual estimates use independent cell halves within each constituent control,
single and pair condition. They do not share the same noisy control/single means.
This is within-screen repeatability, not independent biological replication.'''),
code('''repeatability = pd.read_csv(ROOT / 'results/residual_repeatability.csv')
display(repeatability.describe())
display(repeatability.nlargest(5, 'residual_rms'))'''),
md('''## Verification and provenance
The checks independently recompute errors from saved predictions, check pair splits,
and verify that model checkpoints were selected using validation results.'''),
code('''if (ROOT / 'data/processed/condition_profiles.npz').exists():
    result = subprocess.run([sys.executable, str(ROOT / 'src/verify_results.py')], check=True, capture_output=True, text=True)
    print(result.stdout)
else:
    print('Numerical verification requires the generated condition profiles. Follow README.md to regenerate them; large arrays are omitted from the small code package.')
display(pd.DataFrame(json.loads((ROOT / 'results/source_manifest.json').read_text())))'''),
md('''## Next experiments
1. Restore the upstream cell-level training protocol, graph scope and loss before comparative claims.
2. Repeat model runs across the fixed pair partitions and independent initialization seeds.
3. Add permutation-invariant set attention and evaluate DepMap features, excluding K562.
4. Retain the prespecified attention-plus-DepMap versus additive comparison and report every configuration.

The preliminary negative result does not establish that embeddings cannot help. It does demonstrate a working,
paired comparison against the strong additive predictor on this dataset.''')]
with tempfile.TemporaryDirectory(prefix='ml4g-interim-kernel-') as folder:
    kernels=Path(folder)/'kernels';spec=kernels/'ml4g-interim';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                              'display_name':'ML4G interim','language':'python'}))
    manager=KernelManager(kernel_name='ml4g-interim',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    try:
        nb=NotebookClient(nb,km=manager,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
nbf.write(nb,ROOT/'Interim_Analysis.ipynb')
print('Executed and saved',ROOT/'Interim_Analysis.ipynb')
