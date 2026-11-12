# Full cell-level GPU experiments

[Open the training notebook in Colab](https://colab.research.google.com/github/micepram/ml4fg-gene-pair-prediction/blob/main/Colab_Training.ipynb).
Select a CUDA GPU and run its cells in order. Google Drive mounting needs your authorization.
The existing interim notebook remains a viewer of measured local pilots.

## What changes

| Component | Interim pilots | This pipeline |
|---|---|---|
| Training targets | 225 condition means | Individual cells from training conditions |
| Inputs | Control mean | One fixed random control cell per perturbed cell; controls paired to themselves |
| Genes predicted | 19,264 | 19,264 |
| Hidden size | 32 | 64 upstream preset; 128 extended preset |
| Epoch budget | 20, early stopping | 20 upstream; 50 extended, best validation epoch retained |
| Loss | Ordinary MSE | Upstream per-condition fourth-power error + 0.1 × squared direction discrepancy; nonzero-gene filters |
| GO graph | 100 observed targets | Official GEARS essential-gene graph (9,853 nodes covered by the supplied GO dictionary) |
| Coexpression | Training condition means | All individual training single/control cells, absolute Pearson >0.4, top21 including self |
| Extensions | GenePT + sum | Native / GenePT / DepMap × sum / Set Transformer attention |
| Replication | One partition, one initialization | Five fixed pair partitions × two seeds |
| Recovery | Final model weights | Latest optimizer/scheduler/scaler/RNG state plus separate best validation weights |

The full upstream model remains pinned at `interim/vendor/GEARS/COMMIT` under its MIT
license. Native + sum calls its unmodified forward. Extensions hook only the post-GO
embedding and perturbation aggregation. Attention uses self-attention and learned-seed
pooling without positional encodings. Fixed external vectors are L2 normalized, with a
gene-present indicator; a learned projection adds them to the native representation.

The **default notebook preset is extended**, not a small architecture pilot. Its six
matched configurations each use 128 hidden dimensions, 50 epochs, Adam 1e-3,
weight decay 5e-4, gradient value clipping at 1, and learning-rate halving every 10 epochs.
It also runs the native upstream-sized reference (64 dimensions, 20 epochs, halving
each epoch) on the same partitions/seeds. That is **70 fits**; the upstream preset alone
has 60 fits. All configurations within a preset share settings, control pairing and
evaluation. Mixed FP16 is the notebook default for T4; FP32 is available. These choices
are fixed in advance, not selected by test performance. Stronger training need not beat
additivity. Completing all fits requires multiple sessions, potentially many GPU hours.

## Data and provenance

The exact 124-pair processed release, 84,143 retained cells and existing 62/31/31 pair
splits are preserved. All 100 singles and control cells are available for training.
Preparation reuses the public downloads documented in `interim/results/source_manifest.json`.
It never reapplies normalization to the supplied continuous log-expression matrix.
It builds an approximately 6.5 GB float32 memory-mapped cell matrix on VM disk, avoiding
millions of dense Python graph objects. GPU coexpression uses training singles/controls
only, matching the upstream graph scope. Since these conditions are shared by all five
partitions, this graph is safely reusable. No pair expression enters it.

Official GEARS graph downloads:

- [Essential gene list](https://dataverse.harvard.edu/api/access/datafile/6934320)
- [Precomputed GO graph](https://dataverse.harvard.edu/api/access/datafile/6934319)

Both downloads have pinned SHA256 checks. GO selection follows upstream target-wise
top21 including self edges. Coexpression ties use stable gene-index order; arithmetic
is FP32 on the GPU, so correlation ties/rounding can differ from a CPU NumPy build.

GenePT uses the authors' [Zenodo 10833191 release](https://zenodo.org/records/10833191),
specifically NCBI gene-description ada-002 vectors (1,536 dimensions). Priors cover all
GO nodes where available, not merely the 100 observed targets. Coverage is measured
and written to `genept.json` and `depmap.json`; missing genes remain explicit.

DepMap uses the labeled **22Q4 Public** [Figshare release](https://doi.org/10.6084/m9.figshare.21637199),
with the official MD5-checked `CRISPRGeneEffect.csv` and `Model.csv`. K562
(`ACH-000551`) and models whose normalized cell-line names contain K562 are excluded
**before** imputation and feature normalization. Every included dependency row must
have model metadata. Missing values are replaced with that gene's mean over the
remaining cell lines; entirely absent genes get zeros plus a missing indicator.
This follows the dependency-profile idea in [MORPH](https://github.com/uhlerlab/MORPH),
not its exact supplied embedding array. The authors' shared processed array has 1,100
coordinates without cell-line labels, so it cannot substantiate the planned K562
exclusion and is not used. Knockout viability profiles remain an imperfect prior for
activation-induced expression changes; attention and bigger models do not resolve that.

Each run records code commit and hashes, upstream commit, input hashes, exact settings,
package versions, GPU, CUDA version and elapsed time. Cache files are local/ignored;
checkpoints and metrics are written under `MyDrive/ML4FG_GPU/runs` by default. No
credentials, source datasets or large checkpoints are committed to GitHub.

## Evaluation and remaining protocol differences

The primary metric remains per-pair MSE on the 1,000 genes selected from controls.
Positive paired improvement is additive MSE minus model MSE on the same pair and genes.
Validation pairs alone select the epoch. Test predictions are made after a fit finishes;
top20 DE subsets are scoring-only and never enter training or model selection.
All optimizer seeds are averaged within split/pair, then repeated held-out appearances
are averaged within each pair before equal-weight summaries and pair bootstrap intervals.
Five full splits cover 98 unique test pairs, not 155 independent pairs. Summaries flag
incomplete configurations; compare methods on matched coverage. Bootstrap intervals
are conditional on fitted models and do not resolve dependence among gene-sharing pairs.

These are full-cell experiments, but not an exact reproduction of every published
GEARS result: the processed dataset and pair partition scheme differ, checkpoint
selection uses the proposal's top1000 primary metric, mixed precision is optional,
and inference has a fixed batch size of one. The upstream forward's coexpression
batching behavior is retained for the native reference and every extension; its gene
edges are not replicated across batch members. Predictions can therefore depend on
batch position. Fixing inference to batch size one avoids arbitrary evaluation batch
composition. This issue merits a separate, explicitly labeled implementation sensitivity
analysis rather than silently changing the comparator.

In eval mode, input expression enters the retained architecture only through the final
additive residual. Prediction at the full control mean is therefore mathematically equal
to averaging predictions over all control cells **at batch size one**. A fixture test
verifies this. This exact mean inference does not replace individual-cell training.

## Sessions, updates and commands

Colab's [official FAQ](https://research.google.com/colaboratory/faq.html) describes variable
GPU availability and runtime limits; a free runtime may last up to 12 hours, often less.
The notebook budgets eight training hours and checkpoints every five minutes and epoch.
Nothing prevents Colab from ending earlier. Reconnect normally and rerun to resume.
There are no keepalive scripts or attempts to bypass usage limits. The default experiment
can require roughly 6–12 GB of persistent Drive space as fits accumulate; the large cell
matrix stays off Drive. Check available storage before a long campaign.

An open GitHub notebook is a snapshot. Pushes do not update its cells or imported modules.
Save edits and reopen it to get changed cells; setup uses `git pull --ff-only` for scripts.
Avoid updating during training. Source/configuration mismatches refuse checkpoint resume;
use a new run directory for material changes. Subprocesses start with fresh Python code.
Do not install the macOS `interim/requirements-lock.txt` in Colab: it would replace the
CUDA PyTorch installation. The Colab requirements leave PyTorch intact.

From the repository root in a CUDA environment:

```bash
python -m pip install -r gpu/requirements-colab.txt
python -m gpu.prepare --device cuda
python -m unittest discover -s gpu/tests -v
python -u -m gpu.run_suite --output /path/to/persistent/runs --profile extended --amp fp16 --max-hours 8
python -m gpu.summarize --runs /path/to/persistent/runs
```

The notebook defaults to physical batch 32. If a GPU cannot fit it, launch a **new matched
suite** at batch 16 (or use FP16); do not vary batch size silently across configurations.
Source hashes and configurations prevent accidental resume with changed training rules.
Latest checkpoints contain optimizer and RNG states; best checkpoints contain selected
model weights. Mid-epoch resume is tested against an uninterrupted run for exact CPU
weight and prediction equality. CUDA reductions can still be nondeterministic; deterministic
algorithms are requested with warnings rather than claiming bitwise GPU reproducibility.

No new performance result is claimed until the corresponding `result.json`, predictions,
history and pair metrics exist. The interim report and previously measured pilot outputs
are preserved until these runs provide evidence for a report revision.
