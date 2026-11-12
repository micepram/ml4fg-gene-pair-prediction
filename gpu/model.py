"""Pinned upstream GEARS, with optional frozen priors and set attention.

Hooks change only the post-GO representation and perturbation aggregation.
The native/sum case executes the upstream forward without either hook.
"""
import importlib.util
from types import SimpleNamespace
import torch
from torch import nn
from .common import INTERIM

spec = importlib.util.spec_from_file_location('pinned_gears_model', INTERIM / 'vendor/GEARS/gears/model.py')
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


class SetAttention(nn.Module):
    """SAB followed by one-seed PMA; no positional encodings or token order."""
    def __init__(self, hidden, heads=4):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.pool = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.seed = nn.Parameter(torch.randn(1, 1, hidden) * .02)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))

    def forward(self, tokens, padding):
        z = self.self_attention(tokens, tokens, tokens, key_padding_mask=padding, need_weights=False)[0]
        z = self.norm1(tokens + z)
        z = self.norm2(z + self.ff(z))
        query = self.seed.expand(len(tokens), -1, -1)
        pooled = self.pool(query, z, z, key_padding_mask=padding, need_weights=False)[0]
        return self.norm3(query + pooled)[:, 0]


class PerturbationModel(nn.Module):
    def __init__(self, args, prior=None, aggregation='sum'):
        super().__init__()
        if aggregation not in ('sum', 'attention'):
            raise ValueError(aggregation)
        self.core = upstream.GEARS_Model(args)
        self.aggregation = aggregation
        self.current_perts = None
        self.global_embeddings = None
        self.register_buffer('prior', prior)
        if prior is not None:
            self.prior_projection = nn.Sequential(nn.Linear(prior.shape[1], args['hidden_size']), nn.LayerNorm(args['hidden_size']))
            self.prior_scale = nn.Parameter(torch.tensor(.2))
        if aggregation == 'attention':
            self.set_attention = SetAttention(args['hidden_size'])
            self.core.pert_fuse.register_forward_pre_hook(self._replace_sum)
        if prior is not None or aggregation == 'attention':
            self.core.sim_layers[-1].register_forward_hook(self._capture_global)

    def _capture_global(self, module, inputs, output):
        if self.prior is not None:
            output = output + self.prior_scale * self.prior_projection(self.prior)
        self.global_embeddings = output
        return output

    def _replace_sum(self, module, inputs):
        # Upstream pert_track contains only non-control cells, in batch order.
        sets = [[p for p in ids if p != -1] for ids in self.current_perts]
        sets = [ids for ids in sets if ids]
        width = max(map(len, sets))
        indices = torch.zeros((len(sets), width), dtype=torch.long, device=self.global_embeddings.device)
        padding = torch.ones_like(indices, dtype=torch.bool)
        for i, ids in enumerate(sets):
            indices[i, :len(ids)] = torch.tensor(ids, device=indices.device)
            padding[i, :len(ids)] = False
        pooled = self.set_attention(self.global_embeddings[indices], padding)
        # Preserve upstream's duplication for BatchNorm with one perturbed cell.
        if len(sets) == 1:
            pooled = pooled.repeat(2, 1)
        return (pooled,)

    def forward(self, data):
        self.current_perts = data.pert_idx
        try:
            return self.core(data)
        finally:
            self.current_perts = None
            self.global_embeddings = None


def make_batch(x, perts, device):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    return SimpleNamespace(x=x.reshape(-1, 1), pert_idx=perts,
                           batch=torch.arange(len(x), device=device).repeat_interleave(x.shape[1]))


def gears_loss(pred, truth, conditions, control, masks, direction_lambda=.1):
    """Upstream loss_fct: per-condition fourth powers + lambda*squared sign gap.

    Compute in FP32 under AMP. The sign term has zero gradient, as upstream.
    Only training-condition nonzero masks may be supplied.
    """
    pred, truth = pred.float(), truth.float()
    loss = pred.sum() * 0
    for condition in sorted(set(conditions)):
        rows = [i for i, c in enumerate(conditions) if c == condition]
        mask = masks[condition]
        p, y = pred[rows][:, mask], truth[rows][:, mask]
        c = control[mask]
        loss = loss + (p - y).pow(4).mean()
        loss = loss + direction_lambda * (torch.sign(y - c) - torch.sign(p - c)).square().mean()
    return loss / len(set(conditions))
