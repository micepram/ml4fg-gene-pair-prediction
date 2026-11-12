"""Aggregate seeds, then repeated appearances of each pair; never pool cells."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(runs):
    frames = []
    for path in sorted(runs.glob('*/pair_metrics.csv')):
        if not (path.parent / 'result.json').exists():
            continue
        frame = pd.read_csv(path)
        # A finished CSV is insufficient: ensure training actually completed.
        result = json.loads((path.parent / 'result.json').read_text())
        if result['state'] == 'complete':
            frames.append(frame)
    if not frames:
        print('No completed full-cell fits yet. Checkpoints are not test results.')
        return
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(runs / 'all_pair_metrics.csv', index=False)
    method = ['profile', 'prior', 'aggregation', 'subset']
    values = ['mse', 'additive_mse', 'paired_improvement', 'pearson_delta', 'residual_correlation']
    seed_means = metrics.groupby(method + ['split', 'condition'], as_index=False)[values].mean()
    pairs = seed_means.groupby(method + ['condition'], as_index=False)[values].mean()
    pairs.to_csv(runs / 'pair_means.csv', index=False)
    seed_means.groupby(method + ['split'], as_index=False)[values].mean().to_csv(runs / 'split_means.csv', index=False)
    rows = []
    rng = np.random.default_rng(104729)
    for keys, frame in pairs.groupby(method):
        differences = frame.paired_improvement.to_numpy()
        boot = differences[rng.integers(len(differences), size=(10000, len(differences)))].mean(axis=1)
        available = metrics
        for key, value in zip(method, keys):
            available = available[available[key] == value]
        completed = available[['split', 'seed']].drop_duplicates()
        row = dict(zip(method, keys))
        row.update(completed_fits=len(completed), planned_fits=10, unique_pairs=len(frame),
                   status='complete' if len(completed) == 10 else 'provisional_incomplete_splits_or_seeds',
                   mean_mse=float(frame.mse.mean()), additive_mse=float(frame.additive_mse.mean()),
                   paired_improvement=float(differences.mean()),
                   ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
                   fraction_pairs_better=float((differences > 0).mean()))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(runs / 'summary.csv', index=False)
    print(summary[summary.subset == 'top1000'].to_string(index=False))
    print('Intervals resample pairs after averaging seeds/splits; conditional on fitted models. '
          'Incomplete coverage is provisional. Pairs sharing genes may remain dependent.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, required=True)
    summarize(p.parse_args().runs)
