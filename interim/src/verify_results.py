"""Independent checks for scientific bookkeeping and metric correctness."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from baselines import correlation

ROOT=Path(__file__).resolve().parents[1]
d=np.load(ROOT/'data/processed/condition_profiles.npz')
idx={c:i for i,c in enumerate(d['conditions'])};means=d['means'];top=d['top1000'];control=means[idx['ctrl']]
splits=json.loads((ROOT/'data/processed/splits.json').read_text())
assert len(idx)==225 and sum('+' in c for c in idx)==124
assert np.array_equal(top,np.argsort(-control,kind='stable')[:1000])
for split in splits:
    tr,va,te=map(set,(split['train'],split['validation'],split['test']))
    assert not (tr&va or tr&te or va&te)
    assert tr|va|te==set(idx)
    assert len(va)==len(te)==31
    assert all(g in tr for c in te|va for g in c.split('+'))
    assert sum('+' in c for c in tr)==62
b=pd.read_csv(ROOT/'results/baseline_pair_metrics.csv')
for row in b.itertuples():
    i=idx[row.condition];a,bb=row.condition.split('+')
    additive=means[idx[a]]+means[idx[bb]]-control
    err=float(np.mean((additive[top]-means[i,top])**2))
    assert np.isclose(row.paired_improvement,err-row.mse,atol=1e-8)
assert np.isnan(correlation(np.zeros(5),np.arange(5)))
assert np.isclose(correlation(np.arange(5),np.arange(5)),1)
for slug in ['gears','gears_genept']:
    path=ROOT/f'results/{slug}_predictions.npz'
    if not path.exists():continue
    pred=np.load(path);metrics=pd.read_csv(ROOT/f'results/{slug}_pair_metrics.csv').set_index('condition')
    run=json.loads((ROOT/f'results/{slug}_run.json').read_text())
    history=pd.read_csv(ROOT/f'results/{slug}_training_history.csv')
    assert run['best_epoch']==int(history.loc[history.validation_mse.idxmin(),'epoch'])
    assert len(pred['conditions'])==31
    for c,p in zip(pred['conditions'],pred['predictions']):
        assert np.isfinite(p).all()
        assert np.isclose(np.mean((p[top]-means[idx[c],top])**2),metrics.loc[c,'mse'],atol=1e-8)
print('PASS: disjoint condition splits, singles availability, control-only readout selection, paired signs, undefined zero-delta correlation, saved-prediction metrics, validation-selected checkpoints.')
