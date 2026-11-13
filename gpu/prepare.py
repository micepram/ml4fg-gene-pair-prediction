"""Prepare disk-backed cell targets, official GO graph, and external priors."""
import argparse
import gc
import hashlib
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from .common import INTERIM, ROOT, atomic_json, sha256

EXTRA_SOURCES = {
    'essential_all_data_pert_genes.pkl': ('https://dataverse.harvard.edu/api/access/datafile/6934320', None),
    'go_essential_all.tar.gz': ('https://dataverse.harvard.edu/api/access/datafile/6934319', None),
    'DepMap22Q4_CRISPRGeneEffect.csv': ('https://api.figshare.com/v2/file/download/38357390', '41c32e0472edbb7da35aa127a0c1f759'),
    'DepMap22Q4_Model.csv': ('https://api.figshare.com/v2/file/download/38466923', '528c0c17d4e7c4c18b88e50add1c1c0f'),
}
OFFICIAL_SHA256 = {
    'essential_all_data_pert_genes.pkl': '46c3dfe354d8ad5c0da22c69f3d0ca451987b1a61ed9d984279b22b9565ff8d7',
    'go_essential_all.tar.gz': '98a14a60e8b76f76fd172570d023a6036775f151604ec01340b3fec7d36693da',
}
PRIMARY_SHA256 = {
    'norman_scfoundation.zip': '11c65bdd626129185b6e940aca83c26af584c709d5089bbbb6faa92cc7d6ae9b',
    'gene2go_all.pkl': 'f145c5e84a53048d87942a417d870a4f2d8db50200b96e492b358c13aba8c771',
    'GenePT_embeddings_v2.zip': '6193575dcbd7bbf214c8ca3eb518cb3d13272443f98ecd6c26402411a7ca745e',
}
MIRROR_BASE = ('https://huggingface.co/datasets/NexusMancer/NexusMancer-bioai-easykit-gears-cache/'
               'resolve/6f5f7fd3f2acecf7553176cf88924542a4cdf1f9/gears_cache/')
FALLBACK_URLS = {name: [MIRROR_BASE + name] for name in OFFICIAL_SHA256}


def download(path, url, md5=None):
    sidecar = path.with_suffix(path.suffix + '.source.json')
    chosen_url = json.loads(sidecar.read_text())['download_url'] if sidecar.exists() else url
    if not path.exists():
        print('Downloading', path.name, flush=True)
        temp = path.with_suffix(path.suffix + '.part')
        last_error = None
        for chosen_url in [url, *FALLBACK_URLS.get(path.name, [])]:
            try:
                with urllib.request.urlopen(chosen_url, timeout=240) as response, temp.open('wb') as f:
                    if response.status != 200:
                        raise urllib.error.URLError(f'HTTP {response.status}')
                    received, last_print = 0, time.monotonic()
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        f.write(block)
                        received += len(block)
                        if time.monotonic() - last_print > 5:
                            print(f'{path.name}: {received/2**20:.0f} MB downloaded', flush=True)
                            last_print = time.monotonic()
                expected = {**OFFICIAL_SHA256, **PRIMARY_SHA256}.get(path.name)
                if expected and sha256(temp) != expected:
                    raise ValueError(f'Checksum mismatch for {path.name}; refusing downloaded data')
                temp.replace(path)
                atomic_json(dict(download_url=chosen_url, canonical_url=url), sidecar)
                break
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                print(f'Download unavailable ({error}); trying a hash-pinned fallback if available.', flush=True)
        else:
            raise RuntimeError(f'No source could supply {path.name}') from last_error
    if md5:
        h = hashlib.md5()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != md5:
            raise ValueError(f'Checksum mismatch: {path}. Remove this invalid download and retry.')
    digest = sha256(path)
    expected = {**OFFICIAL_SHA256, **PRIMARY_SHA256}
    if path.name in expected and digest != expected[path.name]:
        raise ValueError(f'Public resource checksum mismatch: {path}')
    return dict(name=path.name, url=chosen_url, canonical_url=url,
                bytes=path.stat().st_size, sha256=digest)


def prepare_cells(cache):
    profiles = np.load(cache / 'profiles.npz')
    conditions = profiles['conditions'].tolist()
    genes = profiles['genes'].tolist()
    a = ad.read_h5ad(INTERIM / 'data/raw/norman.h5ad', backed='r')
    def canonical(condition):
        genes = sorted(g for g in str(condition).split('+') if g != 'ctrl')
        return '+'.join(genes) if genes else 'ctrl'
    lookup = {c: i for i, c in enumerate(conditions)}
    labels = np.array([lookup.get(canonical(c), -1) for c in a.obs.condition], dtype=np.int64)
    keep = labels >= 0
    assert a.var.gene_name.astype(str).tolist() == genes
    path = cache / 'cells.npy'
    if not path.exists():
        temp = cache / 'cells.partial.npy'
        out = np.lib.format.open_memmap(temp, mode='w+', dtype='float32', shape=(int(keep.sum()), len(genes)))
        cursor = 0
        for start in range(0, a.n_obs, 512):
            stop = min(start + 512, a.n_obs)
            x = a.X[start:stop][keep[start:stop]].toarray().astype('float32')
            if not np.isfinite(x).all() or (x < 0).any():
                raise ValueError('Unexpected expression scale')
            out[cursor:cursor + len(x)] = x
            cursor += len(x)
            if start % 8192 == 0:
                print(f'Writing individual-cell cache: {stop}/{a.n_obs}', flush=True)
        out.flush()
        del out
        temp.replace(path)
    np.save(cache / 'labels.npy', labels[keep])
    a.file.close()
    atomic_json(dict(cells=int(keep.sum()), genes=genes, conditions=conditions,
                     expression='Supplied nonnegative log-expression; no second normalization',
                     raw_sha256=sha256(INTERIM / 'data/raw/norman.h5ad')),
                cache / 'cells.json')
    return profiles


def prepare_profiles(cache):
    """Keep generated metadata out of the tracked interim report files."""
    if (cache / 'profiles.npz').exists() and (cache / 'top20.npy').exists() and (cache / 'splits.json').exists():
        return
    source = INTERIM / 'data/processed'
    if not (source / 'condition_profiles.npz').exists():
        build = cache / 'profile_build'
        (build / 'data/raw').mkdir(parents=True, exist_ok=True)
        for name in ['norman.h5ad', 'gene2go_all.pkl']:
            link = build / 'data/raw' / name
            if not link.exists():
                link.symlink_to((INTERIM / 'data/raw' / name).resolve())
        spec = importlib.util.spec_from_file_location('profile_preparation', INTERIM / 'src/prepare_data.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = build
        module.main()
        source = build / 'data/processed'
        expected = json.loads((INTERIM / 'data/processed/splits.json').read_text())
        if json.loads((source / 'splits.json').read_text()) != expected:
            raise ValueError('Generated splits differ from the prespecified project partitions')
    for name, target in [('condition_profiles.npz', 'profiles.npz'), ('splits.json', 'splits.json'),
                          ('top20_de_scoring_only.npy', 'top20.npy')]:
        shutil.copyfile(source / name, cache / target)


def top_go_edges(csv_path, names, k=20):
    """Exactly the upstream target-wise nlargest(k+1), including self edges.

    Stream the ~12M-row table instead of holding all string columns in RAM.
    Stable sorting preserves original file order at tied scores.
    """
    best = None
    for chunk in pd.read_csv(csv_path, chunksize=250_000):
        chunk = chunk[chunk.source.isin(names) & chunk.target.isin(names)]
        candidates = chunk if best is None else pd.concat([best, chunk], ignore_index=True)
        best = (candidates.sort_values(['target', 'importance'], ascending=[True, False], kind='stable')
                .groupby('target', sort=False).head(k + 1))
    index = {g: i for i, g in enumerate(names)}
    edge = np.array([best.source.map(index), best.target.map(index)], dtype=np.int64)
    return edge, best.importance.to_numpy(dtype='float32')


def coexpression(cells, labels, conditions, device, k=20, threshold=.4):
    """All training singles + controls, matching upstream graph scope.

    No pair expression enters this graph. Since every split observes all
    singles, the graph is identical across the five fixed pair partitions.
    """
    selected = np.flatnonzero(np.isin(labels, [i for i, c in enumerate(conditions) if '+' not in c]))
    x = torch.empty((len(selected), cells.shape[1]), dtype=torch.float32, device=device)
    for start in range(0, len(selected), 512):
        x[start:start + 512] = torch.from_numpy(np.array(cells[selected[start:start + 512]], copy=True)).to(device)
    x -= x.mean(dim=0)
    x /= torch.linalg.vector_norm(x, dim=0).clamp_min(1e-12)
    sources, targets, weights = [], [], []
    for start in range(0, x.shape[1], 256):
        corr = (x[:, start:start + 256].T @ x).abs().clamp_max(1)
        # Stable index ordering gives reproducible ties. Includes self, as upstream.
        ids = torch.argsort(corr, dim=1, descending=True, stable=True)[:, :min(k + 1, x.shape[1])]
        values = corr.gather(1, ids)
        rows = torch.arange(start, start + len(corr), device=device)[:, None].expand_as(ids)
        take = values > threshold
        sources.extend(ids[take].cpu().tolist())
        targets.extend(rows[take].cpu().tolist())
        weights.extend(values[take].cpu().tolist())
        if start % 2048 == 0:
            print(f'Cell coexpression: {start}/{x.shape[1]} genes', flush=True)
    del x
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return np.array([sources, targets], dtype=np.int64), np.array(weights, dtype='float32')


def prepare_graphs(cache, profiles, device):
    raw = INTERIM / 'data/raw'
    graph = cache / 'graphs.npz'
    if graph.exists():
        return np.load(graph)['pert_genes'].tolist()
    with (raw / 'essential_all_data_pert_genes.pkl').open('rb') as f:
        essential = pickle.load(f)
    with (raw / 'gene2go_all.pkl').open('rb') as f:
        gene2go = pickle.load(f)
    names = sorted({g for g in essential if g in gene2go})
    targets = {g for c in profiles['conditions'] for g in c.split('+') if g != 'ctrl'}
    if not targets <= set(names):
        raise ValueError('Official default GO graph does not cover all targets')
    csv = cache / 'go_essential_all.csv'
    if not csv.exists():
        with tarfile.open(raw / 'go_essential_all.tar.gz') as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith('go_essential_all.csv'))
            with tar.extractfile(member) as src, csv.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    ge, gw = top_go_edges(csv, names)
    cells = np.load(cache / 'cells.npy', mmap_mode='r')
    ce, cw = coexpression(cells, np.load(cache / 'labels.npy'), profiles['conditions'].tolist(), device)
    np.savez_compressed(graph, pert_genes=np.array(names), ce=ce, cw=cw, ge=ge, gw=gw)
    atomic_json(dict(perturbation_nodes=len(names), go_edges=len(gw), coexpression_edges=len(cw),
                     go='Official GEARS essential_all graph; target-wise top21, self edges retained',
                     coexpression='Individual training single/control cells; absolute Pearson >0.4; target-wise top21',
                     used_conditions=[c for c in profiles['conditions'].tolist() if '+' not in c],
                     pair_expression_used=False), cache / 'graphs.json')
    return names


def prior_matrix(values, genes):
    """Frozen gene vectors, L2 normalized; missingness is an explicit feature."""
    dim = len(next(iter(values.values())))
    x = np.zeros((len(genes), dim), dtype='float32')
    present = np.zeros(len(genes), dtype=bool)
    for i, g in enumerate(genes):
        if g in values:
            v = np.asarray(values[g], dtype='float32')
            finite = np.isfinite(v)
            if finite.any():
                x[i] = np.where(finite, v, v[finite].mean())
                present[i] = True
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return np.column_stack([x, present.astype('float32')]), present


def exclude_k562(frame, metadata):
    metadata = metadata.set_index('ModelID')
    if not frame.index.is_unique or not frame.index.isin(metadata.index).all():
        raise ValueError('Every DepMap profile needs unambiguous model metadata')
    names = metadata.reindex(frame.index).CellLineName.fillna('').astype(str)
    normalized = names.str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    excluded = frame.index[normalized.str.contains('K562', regex=False)].tolist()
    if 'ACH-000551' not in excluded:
        raise ValueError('Expected K562 model ACH-000551 was not identified; refusing features')
    return frame.drop(index=excluded), excluded


def prepare_priors(cache, genes, variants=('depmap',)):
    raw = INTERIM / 'data/raw'
    targets = {g for c in json.loads((cache / 'cells.json').read_text())['conditions'] for g in c.split('+') if g != 'ctrl'}
    coverage = {}
    for variant in variants:
        path = cache / f'{variant}.npz'
        if path.exists():
            coverage[variant] = json.loads(path.with_suffix('.json').read_text())
            continue
        print('Preparing frozen', variant, 'features', flush=True)
        if variant == 'genept':
            with zipfile.ZipFile(raw / 'GenePT_embeddings_v2.zip') as z:
                with z.open('GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle') as f:
                    values = pickle.load(f)
            source = dict(release='Zenodo 10833191; NCBI descriptions; ada-002; 1536 dimensions')
        else:
            frame = pd.read_csv(raw / 'DepMap22Q4_CRISPRGeneEffect.csv', index_col=0)
            frame, excluded = exclude_k562(frame, pd.read_csv(raw / 'DepMap22Q4_Model.csv'))
            symbols = frame.columns.str.replace(r' \(\d+\)$', '', regex=True)
            frame.columns = symbols
            values = frame.T.groupby(level=0, sort=False).mean().T.to_dict('list')
            source = dict(release='DepMap 22Q4 Public; Chronos CRISPR gene effect; Figshare 21637199',
                          excluded_model_ids=excluded, remaining_model_ids=frame.index.tolist(),
                          missing_entries='Per-gene mean over remaining models; all-missing genes -> zeros')
            del frame
        x, present = prior_matrix(values, genes)
        del values
        gc.collect()
        np.savez_compressed(path, genes=np.array(genes), features=x, present=present)
        info = dict(**source, graph_nodes=len(genes), covered_nodes=int(present.sum()),
                    covered_targets=sum(bool(present[genes.index(g)]) for g in targets), targets=len(targets),
                    missing_targets=sorted(g for g in targets if not present[genes.index(g)]),
                    input_dimensions=x.shape[1], normalization='Per-gene L2 plus gene-present indicator; frozen')
        atomic_json(info, path.with_suffix('.json'))
        coverage[variant] = info
    return coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=ROOT / 'gpu/cache')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--priors', nargs='+', choices=['depmap', 'genept'], default=['depmap'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.cache.mkdir(parents=True, exist_ok=True)
    raw = INTERIM / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    primary = json.loads((INTERIM / 'results/source_manifest.json').read_text())
    records = [download(raw / r['name'], r['url']) for r in primary
               if r['name'] != 'GenePT_embeddings_v2.zip' or 'genept' in args.priors]
    h5 = raw / 'norman.h5ad'
    with zipfile.ZipFile(raw / 'norman_scfoundation.zip') as z:
        member = next(n for n in z.namelist() if n.endswith('.h5ad'))
        if not h5.exists() or h5.stat().st_size != z.getinfo(member).file_size:
            print('Extracting Norman AnnData', flush=True)
            temp = h5.with_suffix('.h5ad.part')
            with z.open(member) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            temp.replace(h5)
    prepare_profiles(args.cache)
    records += [download(raw / name, url, md5) for name, (url, md5) in EXTRA_SOURCES.items()
                if 'DepMap' not in name or 'depmap' in args.priors]
    atomic_json(records, args.cache / 'source_manifest.json')
    profiles = prepare_cells(args.cache)
    names = prepare_graphs(args.cache, profiles, args.device)
    coverage = prepare_priors(args.cache, names, args.priors)
    atomic_json(dict(ready=True, coverage=coverage, graph_sha256=sha256(args.cache / 'graphs.npz')),
                args.cache / 'READY.json')
    print('Full cell-level training inputs ready:', args.cache, flush=True)


if __name__ == '__main__':
    main()
