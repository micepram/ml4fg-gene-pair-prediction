# Completed GPU experiment

Both models completed five full epochs on a Tesla T4, using split 101 and seed 17.
The complete timed comparison took **109 minutes 6 seconds**, including GPU checks,
validation, checkpoints and scoring. This is below the shared three-hour limit.
No further training was launched during verification or report updating.

Primary errors on the same 31 held-out gene pairs, using the fixed top 1,000
control-expressed genes (lower is better):

| Predictor | Test MSE | Pairs beating additivity |
|---|---:|---:|
| Additive | **0.010982** | Reference |
| Cell-level GEARS | 0.013747 | 7/31 |
| GEARS + DepMap + attention | 0.019847 | 6/31 |

Validation selected epoch 3 for both neural models. Mean paired improvements
(additive MSE minus model MSE) are -0.002765 for GEARS and -0.008865 for the
extension. Their conditional 95% pair-bootstrap intervals are approximately
[-0.005258, -0.000051] and [-0.013373, -0.004293], respectively.
Additivity also has lower error on the all-gene and top-20-DE subsets.

The extension did not help in this experiment. It changes prior and aggregation
jointly, so their individual contributions cannot be separated here. One split,
one seed and five epochs do not establish that biological embeddings or attention
are generally ineffective. The intervals do not account for partition, training-seed
or gene-sharing dependence. Early validation instability and the retained upstream
batch-dependent graph implementation are documented limitations.

All 27 exported files passed their hashes. Independent NumPy calculations from
the saved predictions reproduce every per-pair error for all three scoring subsets.
The checks confirm matched settings, validation-only checkpoint selection, completed
epochs and the elapsed-time cap. Exported condition means and gene ordering exactly
match the local processed dataset. See `gpu/results/bounded_3h/verification.json`.

The updated three-page report is `Interim_Report.pdf`; its editable source is
`Interim_Report.tex`. `gpu/GPU_Results_Analysis.ipynb` is an executed analysis notebook
that does not train models. The submission ZIP includes source, figures, metadata,
predictions and compact scoring context, so numerical verification needs no GPU.
Raw AnnData, large model checkpoints and the cell cache are omitted. Checkpoints
remain in `MyDrive/ML4FG_GPU/bounded_3h`.
