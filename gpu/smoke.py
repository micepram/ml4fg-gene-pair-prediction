"""Real-dimension CUDA optimizer checks, separate from scientific results."""
import argparse
import gc
import time
from pathlib import Path
import torch
from .common import ROOT, atomic_json
from .data import CellData
from .model import PerturbationModel, make_batch, gears_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT / 'gpu/cache')
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This check requires the real CUDA runtime')
    data = CellData(args.cache, 101)
    x, y, names, perts = data.inputs(data.train_rows[:args.batch_size], data.control_mapping(17))
    masks = data.masks('cuda')
    ctrl = torch.from_numpy(data.control).cuda()
    rows = []
    for hidden, prior, aggregation in [(64, 'native', 'sum'), (128, 'depmap', 'attention'), (128, 'genept', 'attention')]:
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(17)
        model = PerturbationModel(data.model_args(hidden, 'cuda'), data.prior(prior), aggregation).cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.float16):
            pred = model(make_batch(x, perts, 'cuda'))
            loss = gears_loss(pred, torch.from_numpy(y).cuda(), names, ctrl, masks)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite full-dimension CUDA loss or gradients')
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize()
        row = dict(hidden=hidden, prior=prior, aggregation=aggregation, physical_batch=args.batch_size,
                   genes=data.cells.shape[1], loss=float(loss), seconds=time.monotonic() - started,
                   peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30, device=torch.cuda.get_device_name())
        rows.append(row)
        print(row, flush=True)
        del model, optimizer, pred, loss
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(rows, args.cache / 'cuda_smoke.json')


if __name__ == '__main__':
    main()
