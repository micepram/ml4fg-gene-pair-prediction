"""Stream the published Norman AnnData; never densify the full cell matrix."""
from pathlib import Path
import hashlib
import json
import pickle
import platform
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]


def canonical(condition):
    genes = sorted(g for g in str(condition).split('+') if g != 'ctrl')
    return '+'.join(genes) if genes else 'ctrl'


def main():
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(ROOT / 'data/raw/norman.h5ad', backed='r')
    gene2go = pickle.load(open(ROOT / 'data/raw/gene2go_all.pkl', 'rb'))
    obs = a.obs.copy()
    obs['canonical_condition'] = obs.condition.map(canonical).astype(str)
    all_conditions = sorted(obs.canonical_condition.unique())
    excluded = [c for c in all_conditions if any(g not in gene2go for g in c.split('+') if g != 'ctrl')]
    conditions = sorted(set(all_conditions) - set(excluded))
    genes = a.var.gene_name.astype(str).to_numpy(dtype=str)
    assert len(genes) == len(set(genes)), 'Duplicate gene symbols require explicit handling'
    idx = {c: i for i, c in enumerate(conditions)}
    labels = np.array([idx.get(c, -1) for c in obs.canonical_condition])
    rng = np.random.default_rng(104729)
    halves = np.full(len(obs), -1, dtype=int)
    for i in range(len(conditions)):
        rows = rng.permutation(np.flatnonzero(labels == i))
        halves[rows] = np.arange(len(rows)) % 2
    n, g = len(conditions), len(genes)
    sums = np.zeros((n, g), np.float64)
    squares = np.zeros_like(sums)
    half_sums = np.zeros((2, n, g), np.float64)
    counts = np.bincount(labels[labels >= 0], minlength=n)
    half_counts = np.stack([np.bincount(labels[halves == h], minlength=n) for h in range(2)])
    nnz, nfinite, negative, max_value = 0, 0, 0, 0.0
    cell_rows = []
    values = []
    for start in range(0, len(obs), 1024):
        end = min(start + 1024, len(obs))
        x = a.X[start:end].tocsr()
        lab = labels[start:end]
        keep = lab >= 0
        x = x[keep]
        lab = lab[keep]
        h = halves[start:end][keep]
        indicator = sparse.csr_matrix((np.ones(len(lab)), (lab, np.arange(len(lab)))), shape=(n, len(lab)))
        sums += (indicator @ x).toarray()
        squares += (indicator @ x.multiply(x)).toarray()
        for k in range(2):
            sub = h == k
            indicator_h = sparse.csr_matrix((np.ones(sub.sum()), (lab[sub], np.arange(sub.sum()))), shape=(n, sub.sum()))
            half_sums[k] += (indicator_h @ x[sub]).toarray()
        nnz += x.nnz
        nfinite += np.isfinite(x.data).sum()
        negative += (x.data < 0).sum()
        max_value = max(max_value, float(x.data.max()))
        sample = x.data[::max(1, len(x.data)//2000)]
        values.extend(sample.tolist())
        local = np.flatnonzero(keep) + start
        cell_rows.append(pd.DataFrame({'cell_id': obs.index[local], 'condition': np.array(conditions)[lab],
                                      'detected_genes': x.getnnz(axis=1), 'mean_log_expression': np.asarray(x.mean(axis=1)).ravel()}))
        print(f'Processed {end:,}/{len(obs):,} cells', flush=True)
    means = sums / counts[:, None]
    variance = np.maximum((squares - counts[:, None] * means**2) / (counts[:, None]-1), 0)
    half_means = half_sums / half_counts[:, :, None]
    control = idx['ctrl']
    top = np.argsort(-means[control], kind='stable')[:1000]
    pca = PCA(n_components=10, svd_solver='full')
    pcs = pca.fit_transform(means[:, top])
    rows = []
    for i, c in enumerate(conditions):
        row = {'condition': c, 'type': 'control' if c == 'ctrl' else ('single' if '+' not in c else 'pair'),
               'n_cells': int(counts[i]), 'n_half0': int(half_counts[0,i]), 'n_half1': int(half_counts[1,i]),
               'mean_log_expression': float(means[i].mean()), 'PC1': float(pcs[i,0]), 'PC2': float(pcs[i,1])}
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'conditions.csv', index=False)
    pd.concat(cell_rows).to_csv(out / 'cells.csv.gz', index=False)
    np.savez_compressed(out / 'condition_profiles.npz', means=means.astype('float32'), variance=variance.astype('float32'),
                        counts=counts, half_means=half_means.astype('float32'), half_counts=half_counts,
                        conditions=np.array(conditions), genes=genes, top1000=top, expression_sample=np.array(values[::10],dtype='float32'))
    # Differential-expression subsets are used for evaluation only, never model features.
    se2 = variance/counts[:, None] + variance[control]/counts[control]
    numerator = means-means[control]
    t = np.divide(numerator, np.sqrt(se2), out=np.zeros_like(numerator), where=se2>0)
    df = np.divide(se2**2, (variance/counts[:,None])**2/(counts[:,None]-1)+(variance[control]/counts[control])**2/(counts[control]-1),
                   out=np.ones_like(se2), where=se2>0)
    pvalues = 2*stats.t.sf(np.abs(t), df)
    top20 = np.argsort(pvalues, axis=1, kind='stable')[:, :20]
    np.save(out / 'top20_de_scoring_only.npy', top20)
    sampled = a.X[:32].copy()
    sampled.data = np.expm1(sampled.data)
    exp_sums = np.asarray(sampled.sum(axis=1)).ravel()
    info = dict(original_cells=a.n_obs, retained_cells=int(counts.sum()), genes=a.n_vars,
                original_conditions=len(all_conditions), retained_conditions=n,
                singles=sum('+' not in c and c!='ctrl' for c in conditions), pairs=sum('+' in c for c in conditions),
                control_cells=int(counts[control]), excluded_conditions=excluded,
                excluded_cells=int((labels<0).sum()), min_cells=int(counts.min()), median_cells=float(np.median(counts)),
                max_cells=int(counts.max()), zero_fraction=float(1-nnz/(counts.sum()*g)),
                nonfinite_values=int(nnz-nfinite), negative_values=int(negative), max_log_value=max_value,
                expm1_library_size_first32=[float(exp_sums.min()),float(np.median(exp_sums)),float(exp_sums.max())],
                pca_variance_ratio=pca.explained_variance_ratio_.tolist(),
                pca_definition='Descriptive PCA of all condition means on 1000 control-high-expression genes; centered, not variance-scaled; not supplied to models',
                python=sys.version, platform=platform.platform(),
                dataset_url='https://api.figshare.com/v2/file/download/44477939',
                gene2go_url='https://raw.githubusercontent.com/biomap-research/scFoundation/aa51ac12ce3c9fd13bc7f918cf70f5f12e10cd37/GEARS/data/gene2go.pkl')
    (out / 'data_summary.json').write_text(json.dumps(info, indent=2))
    splits=[]
    pairs=np.array([c for c in conditions if '+' in c])
    singles=[c for c in conditions if '+' not in c]
    for seed in [101,202,303,404,505]:
        perm=np.random.default_rng(seed).permutation(pairs)
        tr=len(pairs)//2; va=(len(pairs)-tr)//2
        split=dict(seed=seed, train=singles+perm[:tr].tolist(), validation=perm[tr:tr+va].tolist(), test=perm[tr+va:].tolist())
        assert not (set(split['train']) & set(split['validation']) or set(split['train']) & set(split['test']) or set(split['validation']) & set(split['test']))
        assert all(g in singles for p in pairs for g in p.split('+'))
        splits.append(split)
    (out/'splits.json').write_text(json.dumps(splits,indent=2))
    a.file.close()
    print(json.dumps(info,indent=2),flush=True)


if __name__ == '__main__':
    main()
