# Full-cell training within three hours

[Open the training notebook in Colab](https://colab.research.google.com/github/micepram/ml4fg-gene-pair-prediction/blob/main/Colab_Training.ipynb).
Use a CUDA GPU and run the cells in order. The current stage replaces the previous
70-fit default with **two matched models and a three-hour combined allowance**.
The interim report and its recorded pilot results are preserved.

## Current experiment

| Setting | Both models |
|---|---|
| Models | Native GEARS + sum; GEARS + frozen DepMap + set attention |
| Training targets | Individual cells; every training cell per epoch |
| Output | All 19,264 measured genes |
| Hidden dimension | 64, the upstream default |
| Data partition / initialization | Existing split101 / seed17, shared |
| Epochs | Same for both, at most 20; chosen using GPU timing only |
| Loss | Upstream per-condition fourth-power error plus 0.1×squared direction discrepancy |
| Optimizer | Adam 1e-3, weight decay 5e-4, learning-rate halving each epoch, value clipping 1 |
| Physical batch / precision | Shared batch 32 / FP16 by default |
| Checkpoint selection | Validation-pair MSE on the fixed control-selected top1000 genes |

This remains full-cell training, unlike the interim condition-mean pilots. Model size,
output genes, graph scope and training samples are not reduced to fit a laptop. The
number of epochs is adapted to measured GPU speed **before validation/test results**,
using the slower measured steps and a 35% timing margin. If even one common epoch per
model cannot fit safely, the larger fits do not start. Attention is the SAB+PMA set
encoder without positional embeddings. Fixed vectors enter through a learned projection.
Native/sum executes the pinned upstream forward without hooks changing its output.

The broader six-configuration, five-split, two-seed design remains the final evaluation
plan; this three-hour stage cannot establish a repeated-split result. A single seed and
partition must be labeled preliminary. Larger 50-epoch/70-fit sweeps are no longer launched.

## Enforcing the total allowance

`gpu.run_suite` records the start and absolute deadline in
`MyDrive/ML4FG_GPU/bounded_3h/budget.json`. The interval is at most 10,800 seconds for
**GPU checks, timing, both fits, validation, checkpointing and scoring together**.
Downloading, data preparation and CPU fixture checks precede this allowance. Pauses
count too: reconnecting or rerunning never moves the saved deadline forward.
Do not change the output folder to obtain another allowance without new authorization.

The supervisor leaves a minute unused, reserves time for checkpointing/scoring, and
requests a clean stop before each model's share expires. It kills an uncooperative
training process group at its cutoff. Each trainer also has its own independent hard
watchdog. A local lock prevents duplicate launchers for the same experiment. Requests
above three hours are rejected. Expired allowances launch no training process.

The common epoch target and available wall-time shares are recorded in the plan. If
training is slower than the timing estimate, a stopped fit is explicitly `time_limited`;
its completed epochs and partial selected epoch are reported. Safe checkpoints survive.
If no usable checkpoint exists, no test score is invented. Time-limited and completed
fits are distinguishable in the output. An expired budget cannot be extended by resume.

## Data, graph and priors

The fixed processed Norman release retains 84,143 cells, 100 singles, 124 pairs, control,
and 19,264 genes. Pair cells stay together; all 100 singles/control are training observations.
The supplied continuous nonnegative log-expression is not normalized a second time.
A roughly 6.5 GB float32 memory-mapped matrix holds cell targets on VM disk.
Generated metadata stays under the ignored cache, avoiding changes to the interim report.
Downloads and extraction use temporary files and published/pinned checksums.

The official GEARS [essential gene list](https://dataverse.harvard.edu/api/access/datafile/6934320)
and [GO table](https://dataverse.harvard.edu/api/access/datafile/6934319) are SHA256 pinned.
The supplied GO dictionary covers 9,853 essential-list nodes. Target-wise top 21 edges
including self match upstream selection. Coexpression uses individual training
single/control cells only: absolute Pearson > 0.4, target-wise top 21 including self.
No pair expression enters this graph; all five original partitions share the singles,
so the graph is safely reusable. GPU FP32 rounding and stable tie ordering can differ
from a NumPy build at exactly tied correlations.

Frozen DepMap features use labeled [22Q4 Public](https://doi.org/10.6084/m9.figshare.21637199)
Chronos gene-effect scores and model metadata, both official MD5 checked. K-562
(`ACH-000551`) and P2UR/K-562 (`ACH-002061`) are removed before imputation/normalization.
The remaining 1,076 cell lines cover 17,453 genes and 96/100 perturbation targets.
Missing targets are CKS1B, MAP4K5, RHOXF2 and SAMD1. Gene vectors are L2 normalized;
missing entries use the gene's remaining-cell-line mean; absent genes get zeros plus a
missingness indicator. Full graph-node and target coverage is written to `depmap.json`.
Knockout viability remains an imperfect prior for activation-induced expression changes.

This implements the dependency-profile idea in [MORPH](https://github.com/uhlerlab/MORPH),
not the exact unlabeled shared array, which cannot substantiate K562 exclusion.
GenePT support remains in the code, but its archive/features are skipped in this stage.
Recorded local release validation is in `validation/depmap22q4_coverage.json`.

## Evaluation and implementation limits

The additive predictor and neural models use the same pair, gene subset and expression
scale. Positive paired improvement is additive MSE minus model MSE. Top 20 DE sets are
scoring-only. Primary output is MSE on the fixed control-selected 1000 genes; all-gene
MSE, control-relative correlation and interaction-residual correlation are secondary.
A single-split pair bootstrap is conditional on these fitted models and does not capture
optimizer, partition or gene-sharing dependence. It does not replace the planned repeats.

The original upstream coexpression batching behavior is retained for both models: its
gene edges are not replicated across batch members, so predictions can depend on batch
position. Inference fixes batch size one. Because the input profile enters only through
the final additive path, prediction at the full control mean exactly equals averaging
individual control-input predictions at batch size one; a test verifies this property.
Training still uses individual cells and fixed unpaired controls. These choices, the
processed 124-pair release, validation metric and FP16 mean this is not a reproduction
of every published GEARS number. The upstream commit/license remain in `interim/vendor`.

## Reproduction and checks

```bash
python -m pip install -r gpu/requirements-colab.txt
python -u -m gpu.prepare --device cuda --priors depmap
python -m unittest discover -s gpu/tests -v
python -u -m gpu.run_suite --output /persistent/ML4FG_GPU/bounded_3h --max-hours 3
```

Keep Colab's existing CUDA PyTorch; do not install the macOS interim lockfile. The
notebook explicitly streams child stdout and stderr, including download progress and
tracebacks, into visible notebook output. Stops retain the last safe checkpoint rather
than silently changing batch size or reducing genes. If batch 32 cannot fit, no full fit
starts; resolve memory settings consistently before launching a new comparison.

Tests check upstream forward/loss/gradient equivalence, attention invariance, train-only
coexpression, exact mean inference, labeled K562 removal, exact CPU resume, budget
persistence, no launch after expiration, actual forced termination of an uncooperative
process, and visibility of child-process errors. These checks cannot guarantee external
network reliability or replace full-size tests on the actual CUDA device. The bounded
runner performs those GPU checks before fitting and includes their time in the allowance.
No model-performance result is claimed without saved predictions, metrics and run metadata.
