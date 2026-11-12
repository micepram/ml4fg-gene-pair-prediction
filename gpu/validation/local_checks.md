# Local validation of the GPU pipeline

Command: `interim/.venv/bin/python -m unittest discover -s gpu/tests -v`

All nine checks passed on CPU before launching the Colab notebook:

1. Native/sum wrapper matches the pinned unmodified upstream forward exactly.
2. Fourth-power/direction loss values and gradients match upstream `loss_fct`.
3. Reordering genes preserves attention predictions; gradients reach attention and
   prior projection weights while external vectors stay frozen.
4. Control-mean inference equals the average over individual control inputs at
   inference batch size one.
5. Individual-cell coexpression matches NumPy correlations; changing all pair
   expression leaves this graph unchanged. Training filters exclude held-out conditions.
6. Streaming GO selection matches upstream target-wise top-k selection.
7. K562 and derivative cell lines are excluded using labeled metadata; missing
   metadata raises an error.
8. Stopping after one optimizer step, saving, then resuming produces exactly the
   same CPU weights, predictions and selected epoch as uninterrupted training.
9. The job plan includes all six configurations, five splits and two seeds, plus
   the upstream-sized reference when using the extended preset.

The notebook's JSON schema and every Python cell also validate/compile locally.
These small fixture checks establish implementation behavior; they do not establish
GPU memory fit, predictive accuracy, or a completed full-data training experiment.
`gpu.smoke` separately checks full-data dimensions and optimizer steps on the actual
CUDA device after preparation, saving its output to the local cache.
