"""Prespecified pair-level evaluation; condition weights, not cell weights."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def correlation(x,y):
    x=x-x.mean();y=y-y.mean()
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den>1e-12 else np.nan


def paired_summary(frame, nboot=10000):
    rows=[]
    for method,df in frame.groupby('method'):
        per_pair=df.groupby('condition')[['mse','paired_improvement']].mean()
        delta=per_pair.paired_improvement.to_numpy()
        rng=np.random.default_rng(1879)
        boot=np.mean(rng.choice(delta,(nboot,len(delta)),replace=True),axis=1)
        rows.append(dict(method=method,unique_test_pairs=len(delta),mean_mse=per_pair.mse.mean(),
                         mean_paired_improvement=delta.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),
                         fraction_pairs_better=float(np.mean(delta>0))))
    return pd.DataFrame(rows)


def main():
    data=np.load(ROOT/'data/processed/condition_profiles.npz')
    means=data['means'];conditions=data['conditions'].tolist();index={c:i for i,c in enumerate(conditions)}
    top=data['top1000'];de=np.load(ROOT/'data/processed/top20_de_scoring_only.npy')
    ctrl=means[index['ctrl']]
    splits=json.loads((ROOT/'data/processed/splits.json').read_text())
    records=[]
    for split in splits:
        train=[index[c] for c in split['train'] if c!='ctrl']
        train_mean=means[train].mean(axis=0)
        for c in split['test']:
            i=index[c];a,b=c.split('+');truth=means[i]
            additive=means[index[a]]+means[index[b]]-ctrl
            mse_add=float(np.mean((truth[top]-additive[top])**2))
            for method,pred in [('Additive',additive),('No change',ctrl),('Training mean',train_mean)]:
                mse=float(np.mean((truth[top]-pred[top])**2))
                records.append(dict(split=split['seed'],condition=c,method=method,mse=mse,l2=float(np.linalg.norm(truth[top]-pred[top])),
                                    paired_improvement=mse_add-mse,pearson_delta=correlation(pred[top]-ctrl[top],truth[top]-ctrl[top]),
                                    mse_top20=float(np.mean((truth[de[i]]-pred[de[i]])**2))))
    frame=pd.DataFrame(records)
    frame.to_csv(ROOT/'results/baseline_pair_metrics.csv',index=False)
    frame.groupby(['split','method']).agg(mean_mse=('mse','mean'),mean_delta=('paired_improvement','mean'),n_pairs=('condition','size')).reset_index().to_csv(ROOT/'results/baseline_split_metrics.csv',index=False)
    summary=paired_summary(frame);summary.to_csv(ROOT/'results/baseline_summary.csv',index=False)
    # Independent halves for every constituent condition, including singles/control.
    half=data['half_means']
    residuals=[]
    for c in conditions:
        if '+' not in c:continue
        a,b=c.split('+');i=index[c]
        r=np.stack([half[h,i,top]-half[h,index[a],top]-half[h,index[b],top]+half[h,index['ctrl'],top] for h in range(2)])
        full=means[i,top]-means[index[a],top]-means[index[b],top]+ctrl[top]
        residuals.append(dict(condition=c,residual_rms=float(np.sqrt(np.mean(full**2))),
                              half_residual_correlation=correlation(r[0],r[1]),
                              half_difference_rms=float(np.sqrt(np.mean((r[0]-r[1])**2)))))
    pd.DataFrame(residuals).to_csv(ROOT/'results/residual_repeatability.csv',index=False)
    assert len(frame)==5*31*3, 'Unexpected benchmark count'
    assert (frame.query('method=="Additive"').paired_improvement==0).all()
    assert frame.query('method=="No change"').pearson_delta.isna().all(), 'Zero delta correlation must be undefined'
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
