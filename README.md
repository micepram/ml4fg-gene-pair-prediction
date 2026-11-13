# Predicting gene-pair responses beyond additivity

Pramika Garg · pg2899 · ML4FG course project

This repository contains the proposal, the interim report, and the completed local experiments on the Norman K562 perturbation dataset. It is the dedicated working copy of the project. The original Obsidian-folder files are preserved.

## Current results

The retained dataset contains 84,143 cells, 19,264 genes, 100 single-gene conditions, 124 gene pairs and controls. Three simple baselines were evaluated over five fixed pair partitions. Across the 98 unique pairs appearing in the test sets, additive prediction has mean MSE 0.00929, training-mean prediction 0.0206 and no-change prediction 0.0364.

The two exploratory neural runs use condition means, 32 hidden units, a restricted perturbation graph, ordinary MSE, and one partition/initialization. They are not reproductions of the full published GEARS training protocol. On the same 31 test pairs, additive prediction has MSE 0.0110, the GEARS architecture pilot 0.0237, and the GenePT extension 0.0239. No Colab results are included or claimed.

## Start here

- [Colab_Training.ipynb](Colab_Training.ipynb): two matched full-cell CUDA models under a **three-hour total training limit**, with persistent Drive checkpoints and a deadline that does not reset on rerun. The larger repeated-split study remains future work.
- [gpu/README.md](gpu/README.md): GPU protocol, external-feature sources, session limits and verification.
- `Interim_Report.pdf` and `Interim_Report.tex`: verified three-page interim report.
- `Project Proposal.tex`: current proposal and prespecified final evaluation plan.
- `interim/Interim_Analysis.ipynb`: executed notebook with figures, measured results and verification.
- `interim/README.md`: environment setup and full experiment commands.
- `interim/REPORT_CHECKLIST.md`: requirement-by-requirement report mapping.
- `interim/results/`: recorded metrics, histories, run settings and source hashes.
- `interim/src/`: download, preparation, evaluation, model-pilot, plotting and verification scripts.

Raw downloads, generated large arrays, graph caches and checkpoints have been copied into this working folder but are ignored by Git. They can also be regenerated from the documented public sources. The Python environment is recreated locally rather than copied, because virtual environments contain machine/path-specific entry points.

## Why use Colab next?

A CUDA GPU can make full single-cell training, larger graph/model settings and repeated model fits more practical. Changing hardware alone does not make the present algorithm more accurate. The next experiment should establish a faithful full GEARS baseline using its original loss and cell-level targets before testing the embedding and attention modifications.

Keep the five existing pair partitions, control-selected readout genes and additive comparator fixed. Use validation conditions for model selection. Repeat neural experiments with independent initialization seeds and save checkpoints/results outside Colab's temporary runtime. Better accuracy, especially beating the additive predictor, is a hypothesis to test rather than an expected outcome.

The interim notebook inspects completed local results. `Colab_Training.ipynb` is the separate CUDA training launcher. Code is hosted at [micepram/ml4fg-gene-pair-prediction](https://github.com/micepram/ml4fg-gene-pair-prediction). An already-open Colab notebook does not update automatically after a GitHub push; reopen it for changed cells and rerun setup to pull scripts.

## Git identity

The parent folder's conditional Git configuration resolves to:

```text
user.name  = Pramika Garg
user.email = micepram@gmail.com
```

This is inherited from `~/.gitconfig-pramika`; the global default identity is unchanged.
