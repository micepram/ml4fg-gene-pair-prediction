# Predicting gene-pair responses beyond additivity

Pramika Garg · pg2899 · ML4FG course project

This repository contains the proposal, the interim report, and the completed local experiments on the Norman K562 perturbation dataset. It is the dedicated working copy of the project. The original Obsidian-folder files are preserved.

## Current results

The retained dataset contains 84,143 cells, 19,264 genes, 100 single-gene conditions, 124 gene pairs and controls. Three simple baselines were evaluated over five fixed pair partitions. Across the 98 unique pairs appearing in the test sets, additive prediction has mean MSE 0.00929, training-mean prediction 0.0206 and no-change prediction 0.0364.

Two cell-level models have now completed five epochs each on a Colab T4 in **109 minutes 6 seconds total**, within the three-hour limit. On the same 31 test pairs (split 101, seed 17), top 1,000-gene MSE is **0.010982 for additivity**, **0.013747 for GEARS**, and **0.019847 for GEARS + DepMap + attention**. Both neural checkpoints were selected at epoch 3 using validation pairs. The additive baseline remains best in this preliminary comparison. Exported predictions reproduce all reported errors independently; see [gpu/results/bounded_3h/verification.json](gpu/results/bounded_3h/verification.json).

The earlier two exploratory neural runs are preserved. They used condition means, 32 hidden units, a restricted perturbation graph and ordinary MSE, with test MSE 0.0237 (GEARS architecture pilot) and 0.0239 (+GenePT). These differ in protocol from the cell-level runs; the comparison does not isolate a hardware effect. Neither experiment establishes that embeddings cannot help generally.

## Start here

- [Colab_Training.ipynb](Colab_Training.ipynb): two matched full-cell CUDA models under a **three-hour total training limit**, with persistent Drive checkpoints and a deadline that does not reset on rerun. The larger repeated-split study remains future work.
- [gpu/README.md](gpu/README.md): GPU protocol, external-feature sources, session limits and verification.
- [gpu/GPU_Results_Analysis.ipynb](gpu/GPU_Results_Analysis.ipynb): executed analysis of the completed GPU fits, without retraining.
- `Interim_Report_readable.pdf` and `Interim_Report_readable.tex`: preferred submission version, with clearer prose and the same verified results, figures and references in three pages.
- `Interim_Report.pdf` and `Interim_Report.tex`: preserved earlier wording of the same report.
- `Project Proposal.tex`: current proposal and prespecified final evaluation plan.
- `interim/Interim_Analysis.ipynb`: executed notebook with figures, measured results and verification.
- `interim/README.md`: environment setup and full experiment commands.
- `interim/REPORT_CHECKLIST.md`: requirement-by-requirement report mapping.
- `interim/results/`: recorded metrics, histories, run settings and source hashes.
- `interim/src/`: download, preparation, evaluation, model-pilot, plotting and verification scripts.

Raw downloads, generated large arrays, graph caches and checkpoints have been copied into this working folder but are ignored by Git. They can also be regenerated from the documented public sources. The Python environment is recreated locally rather than copied, because virtual environments contain machine/path-specific entry points.

## GPU follow-up

A CUDA GPU enabled the completed cell-level fits with the upstream loss, broader GO graph and attention/DepMap extension. Changing hardware alone does not make an algorithm more accurate. Longer training, independent seeds and the planned repeated partitions remain future work; no additional training is authorized by the completed three-hour experiment.

Keep the five existing pair partitions, control-selected readout genes and additive comparator fixed. Use validation conditions for model selection. Repeat neural experiments with independent initialization seeds and save checkpoints/results outside Colab's temporary runtime. Better accuracy, especially beating the additive predictor, is a hypothesis to test rather than an expected outcome.

The interim notebook inspects completed local results. `Colab_Training.ipynb` is the separate CUDA training launcher. Code is hosted at [micepram/ml4fg-gene-pair-prediction](https://github.com/micepram/ml4fg-gene-pair-prediction). An already-open Colab notebook does not update automatically after a GitHub push; reopen it for changed cells and rerun setup to pull scripts.

## Git identity

The parent folder's conditional Git configuration resolves to:

```text
user.name  = Pramika Garg
user.email = micepram@gmail.com
```

This is inherited from `~/.gitconfig-pramika`; the global default identity is unchanged.
