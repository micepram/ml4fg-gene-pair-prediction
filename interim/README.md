# Norman gene-pair prediction: interim experiments

Author: Pramika Garg, pg2899.

This folder preserves the earlier local stage, including the condition-mean pilots.
The updated report also includes the completed cell-level GPU experiment; see
`../GPU_Results_Summary.md` and `../gpu/GPU_Results_Analysis.ipynb`. Its result
verification does not require retraining or access to a GPU.

This folder contains the actual data analysis, baselines, two exploratory model runs, figures, and results underlying the interim report. The original proposal files in the parent folder are not modified by these experiments.

## Reproduce locally

Run from the repository root. Python 3.11 is recommended. `uv` can be replaced with standard `venv`/`pip`.

```bash
uv venv --python 3.11 interim/.venv
uv pip install --python interim/.venv/bin/python -r interim/requirements-lock.txt
interim/.venv/bin/python interim/src/download_data.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 interim/.venv/bin/python interim/src/prepare_data.py
interim/.venv/bin/python interim/src/download_data.py --extract-genept
interim/.venv/bin/python interim/src/baselines.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 interim/.venv/bin/python interim/src/train_pilot.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 interim/.venv/bin/python interim/src/train_pilot.py --embedding
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 interim/.venv/bin/python interim/src/make_figures.py
interim/.venv/bin/python interim/src/verify_results.py
```

`download_data.py` records source URLs, file sizes and SHA-256 hashes. The isolated environment and raw downloads are not needed to read the saved results. All local paths resolve relative to this folder, not the current shell directory.

## Files to inspect

- `data/processed/data_summary.json`: actual dataset and normalization checks.
- `data/processed/conditions.csv`: cells per condition and descriptive PCA coordinates.
- `data/processed/splits.json`: five fixed, disjoint pair partitions.
- `results/baseline_pair_metrics.csv`: all 155 test appearances per baseline.
- `results/baseline_summary.csv`: equal-weight summaries over 98 unique tested pairs.
- `results/common_split_comparison.csv`: all models on the same 31 split-101 test pairs.
- `results/gears_run.json`, `results/gears_genept_run.json`: model settings and measured results.
- `results/*training_history.csv`: training and validation curves for checkpoint selection.
- `results/embedding_coverage.json`: published GenePT feature variant and coverage.
- `results/residual_repeatability.csv`: independent-cell-half residual diagnostics.
- `figures/`: original vector PDF figures and PNG previews.
- `vendor/GEARS/`: upstream model source and license; exact revision in `COMMIT`.

## Evaluation definition

The primary task predicts unseen pairs of genes observed as single perturbations during training. Canonical unordered pair identities prevent reversed-label leakage. All single conditions and controls are available for training. Each split contains 62 training, 31 validation and 31 test pairs, using partition seeds 101, 202, 303, 404 and 505. All cells from one condition stay together. Model initialization seed 17 is distinct from partition seeds.

Predictions and targets are means of the supplied normalized/log expression. No second normalization is applied. Primary MSE uses the 1,000 genes with highest mean expression in controls. The additive prediction is `single_A + single_B - control`. The training-mean baseline gives each training perturbation condition equal weight, excluding the control condition.

Paired improvement is `MSE_additive - MSE_model`, so positive values favor the model. Repeated held-out appearances are averaged per unique pair before bootstrap summaries. Intervals use 10,000 bootstrap draws and describe the sampled pairs conditional on fitted predictors. They do not treat repeated partitions or constituent genes as independent biological replication. All five partitions together test 98 of 124 unique pairs.

PCA is descriptive and uses all condition means. Its scores/loadings are not supplied to prediction models. Differential-expression subsets and split-cell residual diagnostics are for scoring/interpretation only. A zero control-relative prediction has undefined Pearson correlation, reported as missing rather than zero.

## Important limit of the model runs

These are **condition-mean GEARS architecture pilots**, not reproductions of published GEARS single-cell training. Both use the upstream `GEARS_Model` class without modifying its source, 32 hidden units, one layer in each graph encoder, all 19,264 readouts, and a GO graph restricted to the 100 target genes. Coexpression is calculated only from training-condition means. The training loss is ordinary MSE across all genes, with Adam (`lr=0.001`, `weight_decay=0.0005`), gradient clipping at 1, batches of four conditions, at most 20 epochs and patience five. Checkpoints minimize validation MSE on the primary 1,000 readouts. Each pilot uses one partition and one initialization.

These choices differ from the full upstream training procedure, including its cell-level targets, broader perturbation graph and fourth-power/direction loss. The report must retain this distinction. The GenePT extension adds a learned linear projection, layer normalization and a learned scale to the post-GO perturbation representations. Published NCBI-description features are fixed and L2-normalized, cover all 100 targets and have 1,536 dimensions. This interim variant does not use UniProt features or DepMap.

The primary hypothesis (attention plus DepMap versus additivity) has now been tested in one completed cell-level GPU comparison, described in `../gpu/README.md`. The additive model performed best in that run. Repeated neural fits across partitions and initializations remain future work; the current comparisons cannot establish a general benefit or lack of benefit from embeddings.

## Submission and sharing

The report, source code and compact results are available in the project repository at https://github.com/micepram/ml4fg-gene-pair-prediction. Executed analysis notebooks are included for inspection, and the separate root `Colab_Training.ipynb` launches GPU training. Large raw downloads and the virtual environment are omitted from the source-code attachment. The documented public source links allow the data to be downloaded again.

## Report source and code package

The preferred submission files are `Interim_Report_readable.pdf` and `Interim_Report_readable.tex` in the parent folder. The earlier `Interim_Report.pdf` and `.tex` are preserved. Both reports use `Interim_Report_References.bib` and the figures `gpu/figures/01_workflow.pdf`, `interim/figures/02_eda.pdf` and `gpu/figures/03_performance.pdf` at those relative paths. Both PDFs have three pages including references. Standard LaTeX/BibTeX or Tectonic can compile the source.

`Interim_Report_Code.zip` is a small source/results package. It includes this folder's scripts, executed notebook, metrics, summaries, split lists and figures, plus the report sources. It includes compact GPU predictions and scoring context for numerical verification without retraining. It omits large raw downloads, other generated profile arrays, graph caches, model checkpoints and the virtual environment. Those can be regenerated with the commands above. In the local workspace, all generated arrays and checkpoints remain available.

Open `Interim_Analysis.ipynb` in Jupyter, VS Code or Colab to inspect its saved outputs. Select the local `.venv` interpreter when executing locally. To regenerate its executed outputs programmatically after running the pipeline, run `interim/.venv/bin/python interim/src/build_notebook.py`; its kernel registration is temporary.
