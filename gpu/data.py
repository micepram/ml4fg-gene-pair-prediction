import json
from pathlib import Path
import numpy as np
import torch


class CellData:
    def __init__(self, cache, split_seed):
        self.cache = Path(cache)
        self.cells = np.load(self.cache / 'cells.npy', mmap_mode='r')
        self.labels = np.load(self.cache / 'labels.npy')
        self.meta = json.loads((self.cache / 'cells.json').read_text())
        self.conditions = self.meta['conditions']
        self.index = {c: i for i, c in enumerate(self.conditions)}
        self.profiles = np.load(self.cache / 'profiles.npz')
        if self.profiles['conditions'].tolist() != self.conditions or self.profiles['genes'].tolist() != self.meta['genes']:
            raise ValueError('Cell cache and condition-profile gene/condition order differ')
        if self.cells.shape != (len(self.labels), len(self.meta['genes'])):
            raise ValueError('Cell cache dimensions differ')
        self.means = self.profiles['means']
        self.top = self.profiles['top1000']
        self.control = self.means[self.index['ctrl']]
        splits = json.loads((self.cache / 'splits.json').read_text())
        self.split = next(s for s in splits if s['seed'] == split_seed)
        tr, va, te = [set(self.split[k]) for k in ('train', 'validation', 'test')]
        if tr & va or tr & te or va & te:
            raise ValueError('Conditions overlap across partitions')
        for pair in va | te:
            if any(g not in tr for g in pair.split('+')):
                raise ValueError('A held-out target lacks an observed single perturbation')
        self.train_rows = np.flatnonzero(np.isin(self.labels, [self.index[c] for c in tr]))
        self.control_rows = np.flatnonzero(self.labels == self.index['ctrl'])
        self.graphs = np.load(self.cache / 'graphs.npz')
        self.pert_genes = self.graphs['pert_genes'].tolist()
        pidx = {g: i for i, g in enumerate(self.pert_genes)}
        self.perts = {c: [pidx[g] for g in c.split('+')] if c != 'ctrl' else [-1] for c in self.conditions}

    def inputs(self, rows, control_mapping):
        names = [self.conditions[i] for i in self.labels[rows]]
        x = np.array(self.cells[control_mapping[rows]], copy=True)
        y = np.array(self.cells[rows], copy=True)
        return x, y, names, [self.perts[c] for c in names]

    def control_mapping(self, seed):
        # One fixed random unpaired control per cell, as in upstream preprocessing.
        mapping = np.random.default_rng(seed + 7919).choice(self.control_rows, len(self.cells))
        mapping[self.control_rows] = self.control_rows
        return mapping

    def masks(self, device):
        # Condition-specific nonzero target means, matching upstream. Never
        # construct validation/test masks for use in the training objective.
        return {c: torch.tensor(np.arange(len(self.control)) if c == 'ctrl' else
                                np.flatnonzero(self.means[self.index[c]] != 0), device=device)
                for c in self.split['train']}

    def prior(self, variant):
        if variant == 'native':
            return None
        p = np.load(self.cache / f'{variant}.npz')
        if p['genes'].tolist() != self.pert_genes:
            raise ValueError('Prior gene order differs from the GO graph')
        return torch.from_numpy(p['features'].copy())

    def model_args(self, hidden, device):
        return dict(num_genes=self.cells.shape[1], num_perts=len(self.pert_genes), hidden_size=hidden,
                    uncertainty=False, num_go_gnn_layers=1, num_gene_gnn_layers=1,
                    decoder_hidden_size=16, no_perturb=False, device=device,
                    G_coexpress=torch.from_numpy(self.graphs['ce']),
                    G_coexpress_weight=torch.from_numpy(self.graphs['cw']),
                    G_go=torch.from_numpy(self.graphs['ge']), G_go_weight=torch.from_numpy(self.graphs['gw']))
