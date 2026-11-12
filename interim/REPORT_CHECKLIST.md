# Interim report requirement mapping

Single author: Pramika Garg (pg2899). The PDF has three pages including references. Main text is 11 pt.

| Requirement | Completed evidence |
| --- | --- |
| Title and team | Title and sole author on page 1. |
| Introduction | Biological question, combination-target motivation, Norman data, graph neural network and embedding approach, and cited related work. |
| Download/load/preprocess | Published AnnData downloaded, backed loading and streaming summaries implemented. GO filtering leaves 84,143 cells, 19,264 genes, 100 singles, 124 pairs and controls. Existing expression scale retained. |
| Data understanding | Cell counts, condition types, coverage imbalance, sparsity, nonzero-expression distribution, PCA and loadings, missing GO annotation, missing-feature check. |
| Methods tried | Three simple baselines over five random partitions, a condition-mean GEARS architecture pilot, and a trained GenePT extension. Model protocol departures are explicit. |
| Preliminary results | Actual pair-level errors, repeated-partition summaries, paired intervals, one-partition model results, and residual repeatability. Negative model result is reported. |
| Next steps | Restore full GEARS training, repeat across partitions and initialization seeds, then attention/DepMap comparisons under the existing evaluation design. |
| References | Five relevant papers, in-text citations and BibTeX records. |
| First figure | Original diagram of the completed workflow. |
| Figures | Three original vector figures with readable axes, captions and interpretations. |
| Scientific writing | Completed work in past/present tense, future work identified, source methods distinguished from the new extension. No assertion of therapeutic efficacy or broad model superiority. |
| Reproducibility | Scripts, executed notebook, exact split lists, saved predictions and metrics, source hashes, pinned upstream model revision and dependency lock. |

## Limits that must remain in the submitted report

- No full reproduction of published single-cell GEARS training has been completed.
- Both neural pilots use one partition and one initialization, with simplified condition-mean training and smaller graph scope/hidden dimension.
- Set attention and DepMap remain future work. The final primary comparison has not been tested.
- The five simple-baseline partitions cover 98 unique test pairs, not all 124 pairs and not 155 independent pairs.
- The repeatability diagnostic uses cell halves from the same screen, not independent biological replicates.
- Code/data have not been published to an external service. The local notebook and reproducibility package accompany the report.

This checklist records evidence against the supplied requirements; it is not a guarantee of a particular grade.
