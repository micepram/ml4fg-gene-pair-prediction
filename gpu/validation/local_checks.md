# Local validation of the GPU pipeline

Command: `interim/.venv/bin/python -m unittest discover -s gpu/tests -v`

The original nine checks passed on CPU before the first Colab setup attempt:

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
9. The original large job plan contained the proposed repeated configurations;
   it has since been replaced by the two-model stage requested by the user.

The current suite has **16 passing CPU checks**. It retains the implementation
checks above and now verifies the two-model job list, distinguishes time-limited
fits from completed fits, preserves an absolute deadline across reruns, rejects
allowances above three hours, launches no process after expiry, actually kills an
uncooperative subprocess at a short test deadline, computes common epochs from
timing only, and streams child stdout/stderr visibly for notebook diagnostics.
The added cold-start preparation test builds profiles and the individual-cell
cache from a small AnnData fixture and checks that source metadata is preserved.

The notebook's JSON schema and every Python cell also validate/compile locally.
These small fixture checks establish implementation behavior; they do not establish
GPU memory fit, predictive accuracy, or a completed full-data training experiment.
`gpu.benchmark` checks full-data dimensions and optimizer steps on the actual CUDA
device inside the shared allowance. The old standalone smoke entry point does
not launch separate, unbudgeted GPU work.
