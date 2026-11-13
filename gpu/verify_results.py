"""Recompute exported prediction errors without importing or fitting a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(root):
    manifest = json.loads((root / 'artifact_manifest.json').read_text())
    for entry in manifest['files']:
        path = root / entry['path']
        data = path.read_bytes()
        assert len(data) == entry['bytes'], entry['path']
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
    d = np.load(root / 'evaluation_context.npz', allow_pickle=False)
    names, genes, means = d['conditions'].tolist(), d['genes'].tolist(), d['means']
    index = {c: i for i, c in enumerate(names)}
    control = means[index['ctrl']]
    top = d['top1000']
    np.testing.assert_array_equal(top, np.argsort(-control, kind='stable')[:1000])
    assert means.shape == (225, 19264) and len(genes) == len(set(genes))
    assert int(d['counts'].sum()) == 84143
    split = next(s for s in json.loads((root / 'data_provenance/splits.json').read_text()) if s['seed'] == 101)
    tr, va, te = [set(split[k]) for k in ['train', 'validation', 'test']]
    assert not (tr & va or tr & te or va & te)
    assert len(te) == 31 and all(g in tr for pair in te for g in pair.split('+'))
    budget = json.loads((root / 'budget.json').read_text())
    stopped = json.loads((root / 'budget_status.json').read_text())
    elapsed = stopped['stopped_epoch'] - budget['started_epoch']
    assert 0 < elapsed <= budget['limit_seconds'] <= 10800
    assert stopped['stopped_epoch'] <= budget['hard_deadline_epoch']
    np.testing.assert_allclose(elapsed, stopped['elapsed_wall_seconds'], atol=1e-6)
    checked = []
    for folder in sorted(root.glob('bounded_*_split101_seed17')):
        info = json.loads((folder / 'run.json').read_text())
        result = json.loads((folder / 'result.json').read_text())
        config = info['config']
        assert result['state'] == 'complete' and result['epochs_completed'] == config['epochs'] == 5
        assert config['hidden'] == 64 and config['batch_size'] == 32 and config['amp'] == 'fp16'
        assert config['split'] == 101 and config['seed'] == 17
        history = pd.read_csv(folder / 'history.csv')
        assert history.epoch.tolist() == [1, 2, 3, 4, 5] and not history.partial_epoch.any()
        selected = history.loc[history.validation_mse.idxmin()]
        assert int(selected.epoch) == result['best_epoch']
        np.testing.assert_allclose(selected.validation_mse, result['validation_mse'], rtol=1e-10)
        p = np.load(folder / 'predictions.npz', allow_pickle=False)
        prediction_names = p['conditions'].tolist()
        assert prediction_names == split['test']
        pred = p['predictions']
        assert pred.shape == (31, 19264) and np.isfinite(pred).all()
        stored = pd.read_csv(folder / 'pair_metrics.csv')
        primary, additive_primary = [], []
        for name, prediction in zip(prediction_names, pred):
            row = index[name]
            a, b = name.split('+')
            additive = means[index[a]] + means[index[b]] - control
            truth = means[row]
            for subset, selected_genes in [('top1000', top), ('all', np.arange(len(genes))),
                                            ('top20DE_scoring_only', d['top20'][row])]:
                error = float(np.mean((prediction[selected_genes].astype('float64') - truth[selected_genes]) ** 2))
                baseline_error = float(np.mean((additive[selected_genes].astype('float64') - truth[selected_genes]) ** 2))
                recorded = stored[(stored.condition == name) & (stored.subset == subset)].iloc[0]
                np.testing.assert_allclose([error, baseline_error, baseline_error-error],
                                          [recorded.mse, recorded.additive_mse, recorded.paired_improvement],
                                          rtol=2e-6, atol=2e-9)
                if subset == 'top1000':
                    primary.append(error)
                    additive_primary.append(baseline_error)
        mse = float(np.mean(primary))
        mse_add = float(np.mean(additive_primary))
        np.testing.assert_allclose(mse, result['test_mse'], rtol=2e-6)
        checked.append(dict(prior=config['prior'], aggregation=config['aggregation'],
                            best_epoch=result['best_epoch'], epochs_completed=5, pairs=31,
                            test_mse=mse, additive_mse=mse_add,
                            pairs_better_than_additive=int(np.sum(np.array(primary) < np.array(additive_primary)))))
    assert len(checked) == 2
    # The two models must share every relevant setting except the prior/aggregator.
    configs = [json.loads((p / 'run.json').read_text())['config'] for p in sorted(root.glob('bounded_*_split101_seed17'))]
    ignore = {'prior', 'aggregation', 'prior_sha256'}
    assert {k: v for k, v in configs[0].items() if k not in ignore} == {k: v for k, v in configs[1].items() if k not in ignore}
    summary = dict(passed=True, archive_files_checked=len(manifest['files']), models=checked,
                   total_wall_seconds=elapsed, total_wall_minutes=elapsed/60, under_three_hours=True,
                   verification='Independent NumPy errors on saved predictions, all three scoring subsets; validation-only selection; matched settings; artifact hashes',
                   limitation='One pair partition, one initialization, five epochs; conditional pair-level results')
    (root / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results/bounded_3h')
    verify(p.parse_args().results)
