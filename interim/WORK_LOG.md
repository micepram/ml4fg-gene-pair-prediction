# Interim-report work log

## Authorized scope

Complete data preparation, EDA, five-split baselines, a GEARS pilot, progress on one embedding extension, reproducible code, and an interim report in LaTeX/PDF. Single author: Pramika Garg, pg2899. Work locally first. User has Colab if needed. No deadline supplied. Keep proposal files unchanged during report work.

## Environment and sources

- Local Apple Silicon, 8 GB RAM, 8 CPUs. Torch MPS available; no NVIDIA CUDA.
- Isolated Python 3.11 environment in `.venv`; dependency snapshot in `requirements-lock.txt`.
- Norman source: Figshare file 44477939 via API endpoint (the public browser download endpoint returned empty HTTP 202).
- Source AnnData: 84,986 cells, 19,264 genes, 227 conditions. GO filtering excludes conditions containing IER5L and leaves 225 conditions.
- Harvard Dataverse GO download returned HTTP 403. Identical-purpose published GO dictionary available in scFoundation GitHub at GEARS/data/gene2go.pkl.
- Upstream GEARS source vendored at commit recorded in `vendor/GEARS/COMMIT`; model imported directly to avoid package-level download side effects.
- GenePT archive downloaded from Zenodo record 10833191. All 100 targets have 1,536-dimensional NCBI-description features.

## Status

- [x] Read course/report requirements and revised proposal.
- [x] Download and inspect dataset; identify the 124-pair release and GO exclusion.
- [x] Install local analysis and Torch/PyG environment.
- [x] Run streaming data preparation and five-partition baselines.
- [x] Generate workflow, EDA, and measured performance figures.
- [x] Obtain GenePT features, measure coverage, and run a GEARS/embedding pilot.
- [x] Write and render a 1–3 page interim report with actual results, 11–12 pt text, and at least three references.
- [x] Verify report numbers against saved results; document reproducibility.

## Scientific constraints

All singles remain available for training. Pair conditions, including all their cells, are split 50/25/25 using fixed seeds 101,202,303,404,505. Primary readout: 1000 highest-mean genes in controls. No test conditions in training-derived graphs or checkpoint selection. Means are of the supplied normalized/log expression, with no second transformation. Repeated test appearances are averaged per unique pair before bootstrap summaries. Diagnostic PCA on all condition means is not supplied as a model feature. Published results are context, never reported as our measurements. A reduced or condition-mean GEARS pilot must be labeled explicitly and cannot be called a full reproduction of published single-cell training.

## Completed measurements

- Retained cells: 84,143. Stored zeros verified absent, so 85.1400% matrix sparsity is an exact stored-matrix calculation.
- Five split baseline summaries cover 98 unique test pairs (155 appearances). Additive MSE 0.00929; training mean 0.0206; control/no change 0.0364. Use saved CSV values for exact numbers.
- Split 101: additive MSE 0.0109819; GEARS condition-mean pilot 0.0236836; GEARS + GenePT pilot 0.0239326. No gain from this embedding pilot.
- Native pilot: 11 epochs, selected epoch 6. GenePT pilot: 14 epochs, selected epoch 9. CPU, 4 threads, 32 hidden units; one initialization per model.
- Median independent-cell-half residual correlation: 0.435112. This is not independent biological replication.
- Data preparation, baseline metric consistency, saved-prediction metrics, validation checkpoint selection and notebook execution verified.
- Root-level Interim_Report.tex and Interim_Report_References.bib generate a three-page report including all figures and references. The report is distinct from the proposal.
- Full original single-cell GEARS training, five-partition/two-seed neural comparisons, set attention and DepMap remain final-project work, not claimed as done.
