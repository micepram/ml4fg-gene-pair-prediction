"""Build an executed, read-only notebook over the verified Colab artifacts."""
from pathlib import Path
import json,sys,tempfile
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
nb.cells=[
md('''# Completed GPU experiment
Pramika Garg · ML4FG

Both cell-level models completed five epochs on split 101/seed 17. The entire timed
comparison took 109 minutes 6 seconds, within the three-hour limit. This notebook
reads saved artifacts; **running it does not start or resume training**.

Additive prediction has the lowest primary error. These are preliminary results
from one split, one initialization and five epochs, not the final repeated comparison.'''),
code('''from pathlib import Path
import json,sys,subprocess
import pandas as pd
from IPython.display import display, Image
ROOT=Path.cwd()
if not (ROOT/'gpu').exists(): ROOT=ROOT.parent
RESULTS=ROOT/'gpu/results/bounded_3h'
verification=subprocess.run([sys.executable,'-m','gpu.verify_results','--results',str(RESULTS)],cwd=ROOT,capture_output=True,text=True,check=True)
print(verification.stdout)
'''),
code('''summary=pd.read_csv(RESULTS/'summary.csv')
primary=summary.query("subset=='top1000'").copy()
rows=[dict(method='Additive',mse=primary.additive_mse.iloc[0],pairs_better_than_additive='-')]
for prior,label in [('native','GEARS'),('depmap','GEARS + DepMap + attention')]:
    row=primary.query('prior==@prior').iloc[0]
    rows.append(dict(method=label,mse=row.mean_mse,pairs_better_than_additive=f'{round(row.fraction_pairs_better*31)}/31'))
display(pd.DataFrame(rows))
display(primary[['prior','mean_mse','paired_improvement','ci_low','ci_high','status']])
'''),
md('''## Fixed training settings and selection
Both models use 64 hidden units, all 19,264 genes, 67,592 training cells, batch 32,
FP16, the same split/initialization, original fourth-power/direction loss and schedule.
Both selected epoch 3 using validation error, then scored the same 31 held-out pairs.
The extension changes prior and aggregation together, so this comparison cannot
isolate their individual contributions. The additive predictor also wins on the
all-gene and top 20-DE scoring subsets.'''),
code('''runs=[]
for name in ['bounded_native_sum_split101_seed17','bounded_depmap_attention_split101_seed17']:
    result=json.loads((RESULTS/name/'result.json').read_text())
    result['method']=name
    runs.append(result)
display(pd.DataFrame(runs)[['method','state','epochs_completed','best_epoch','validation_mse','test_mse','elapsed_seconds','skipped_amp_updates']])
display(summary[['prior','subset','mean_mse','additive_mse','paired_improvement']])
'''),
code("display(Image(filename=str(ROOT/'gpu/figures/03_performance.png')))"),
code("display(Image(filename=str(ROOT/'gpu/figures/04_training.png')))"),
md('''## Interpretation and limits
The neural configurations did not beat additivity here. The extension's validation
error was unstable early in training, so its negative result is not evidence that
all embeddings or attention are ineffective. Additional splits, seeds, schedules
and separate prior/aggregation ablations remain future work. The upstream graph's
batch dependence is documented in the protocol; this is not a reproduction of every
published GEARS number. Pair-bootstrap intervals are conditional on these fitted
models and do not capture split/seed or gene-sharing dependence.

A few FP16 updates were skipped by the gradient scaler, as recorded in the run
metadata. This is handled scaler behavior, not a failed fit. CUDA attention warns
that GPU arithmetic may not be bitwise deterministic. No further training was
performed after these artifacts were collected.''')]
with tempfile.TemporaryDirectory(prefix='ml4fg-results-kernel-') as temporary:
    kernels=Path(temporary)/'kernels';spec=kernels/'ml4fg-results';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='ML4FG results',language='python')))
    manager=KernelManager(kernel_name='ml4fg-results',transport='ipc',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    try:
        nb=NotebookClient(nb,km=manager,timeout=120,resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
nbf.write(nb,ROOT/'gpu/GPU_Results_Analysis.ipynb')
print('Saved executed GPU_Results_Analysis.ipynb')
