# Interim report requirement mapping

Single author: Pramika Garg (pg2899). The updated PDF has three pages including
references, with 11 pt main text. This is evidence against the supplied requirements,
not a guarantee of a grade.

| Requirement | Completed evidence |
|---|---|
| Title and team | Title and sole author on page 1. |
| Introduction | Biological question, combination-target motivation, Norman data, GEARS, biological features and attention, with cited related work. |
| Download/load/preprocess | Published AnnData loaded and summarized; 84,143 retained cells, 19,264 genes, 100 singles, 124 pairs and controls. Supplied expression scale retained. |
| Data understanding | Cell-count imbalance, sparsity, expression distribution, PCA, loadings and missing GO/DepMap features. |
| Methods tried | Three simple baselines over five pair partitions; two completed full-cell models using the upstream loss, 64 hidden units and broad GO graph. Earlier GEARS/GenePT mean-based pilots remain available. |
| Preliminary results | Actual pair-level errors, conditional pair intervals, five epochs per neural model, validation-selected checkpoints, and residual repeatability. Negative model results are reported. |
| Next steps | Repeat five partitions/two seeds, test longer schedules and separate embedding from aggregation effects. Investigate validation instability and upstream batching. |
| References | Six relevant papers/preprints, in-text citations and BibTeX records; linked DepMap data release. |
| Figures | Three original vector figures: workflow, EDA and paired results. |
| Writing | Completed versus future work distinguished; no therapeutic-efficacy claim or blanket claim against embeddings. |
| Reproducibility | Source code, saved metrics/predictions, scoring context, executed analysis notebook, split lists, hashes, package versions and pinned upstream revision. |

## Limits that remain explicit

- The GPU comparison uses one partition, one seed and five epochs, not the full repeated study.
- Full-cell targets, upstream loss and broader graph support are implemented, but dataset,
  validation metric, precision and inference differ from published protocols. No exact
  reproduction of every GEARS paper number is claimed.
- Native/sum versus DepMap/attention changes two components jointly. Separate ablations
  are needed to attribute effects to the prior or aggregation.
- Additivity is best on the 31 shared test pairs; this does not establish that all
  embeddings or attention methods are ineffective.
- The five simple-baseline partitions cover 98 unique pairs, not 155 independent pairs.
- Cell-half repeatability is within one screen, not independent biological replication.
- Both GPU fits finished in 109 minutes 6 seconds, below the three-hour cap. No further
  training was performed during result verification or report updating.
- Code and small results are on GitHub. Public raw data and large model checkpoints
  are omitted from the submission package; the compact scoring context and predictions
  are included so numerical verification needs no retraining.
