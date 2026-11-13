"""Full-dimension GPU checks and timing; consumes the shared allowance."""
import argparse
import gc
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, atomic_json
from .data import CellData
from .model import PerturbationModel, make_batch, gears_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT / 'gpu/cache')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--amp', choices=['off', 'fp16'], default='fp16')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; no full training was started')
    torch.set_num_threads(2)
    data = CellData(args.cache, 101)
    masks = data.masks('cuda')
    control = torch.from_numpy(data.control).cuda()
    mapping = data.control_mapping(17)
    rng = np.random.default_rng(17)
    rows = []
    for prior, aggregation in [('native', 'sum'), ('depmap', 'attention')]:
        torch.manual_seed(17)
        torch.cuda.reset_peak_memory_stats()
        model = PerturbationModel(data.model_args(64, 'cuda'), data.prior(prior), aggregation).cuda().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0005)
        times = []
        for step in range(6):
            indices = rng.choice(data.train_rows, args.batch_size, replace=False)
            torch.cuda.synchronize()
            start = time.monotonic()
            x, y, names, perts = data.inputs(indices, mapping)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp == 'fp16'):
                predictions = model(make_batch(x, perts, 'cuda'))
                loss = gears_loss(predictions, torch.from_numpy(y).cuda(), names, control, masks)
            loss.backward()
            if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('GPU check found nonfinite loss/gradients; full fits were not started')
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
            optimizer.step()
            torch.cuda.synchronize()
            if step:
                times.append(time.monotonic() - start)
        row = dict(prior=prior, aggregation=aggregation, hidden=64, batch_size=args.batch_size,
                   seconds_per_step=float(max(times)), peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30,
                   finite_loss_and_gradients=True)
        rows.append(row)
        print('GPU check/timing:', row, flush=True)
        del model, optimizer, predictions, loss
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(dict(models=rows, steps_per_epoch=len(data.train_rows) // args.batch_size,
                     device=torch.cuda.get_device_name(), torch=torch.__version__,
                     warning='Disposable timing updates, not model performance results'), args.output)


if __name__ == '__main__':
    main()
