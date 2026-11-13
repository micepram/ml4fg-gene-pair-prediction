"""Full-cell training with validation-only selection and resumable checkpoints."""
import argparse
import importlib.metadata
import json
import math
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import ROOT, INTERIM, atomic_json, atomic_save, load_checkpoint, provenance, sha256
from .data import CellData
from .model import PerturbationModel, make_batch, gears_loss

STOP = False


def stop_requested(signum, frame):
    global STOP
    STOP = True


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def predict_means(model, data, names, device):
    """Fixed batch-one inference, without using held-out target cell inputs.

    Upstream GEARS adds input expression only at its output. In eval mode,
    prediction at the control mean therefore equals the mean over every
    control cell evaluated at batch size one. No Monte Carlo approximation.
    """
    model.eval()
    return np.stack([model(make_batch(data.control[None], [data.perts[c]], device)).float().cpu().numpy()[0]
                     for c in names])


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 1e-12 else np.nan


def score_test(model, data, config, out, device):
    names = data.split['test']
    predictions = predict_means(model, data, names, device)
    top20 = np.load(data.cache / 'top20.npy')
    rows = []
    for c, pred in zip(names, predictions):
        i = data.index[c]
        a, b = c.split('+')
        additive = data.means[data.index[a]] + data.means[data.index[b]] - data.control
        truth = data.means[i]
        for subset, genes in [('top1000', data.top), ('all', np.arange(len(truth))), ('top20DE_scoring_only', top20[i])]:
            mse = float(np.mean((pred[genes] - truth[genes]).astype('float64') ** 2))
            base = float(np.mean((additive[genes] - truth[genes]).astype('float64') ** 2))
            rows.append(dict(split=config['split'], seed=config['seed'], prior=config['prior'],
                             aggregation=config['aggregation'], profile=config['profile'], condition=c, subset=subset,
                             mse=mse, additive_mse=base, paired_improvement=base - mse,
                             pearson_delta=correlation(pred[genes] - data.control[genes], truth[genes] - data.control[genes]),
                             residual_correlation=correlation(pred[genes] - additive[genes], truth[genes] - additive[genes])))
    pd.DataFrame(rows).to_csv(out / 'pair_metrics.csv', index=False)
    np.savez_compressed(out / 'predictions.npz', conditions=np.array(names), predictions=predictions)
    return float(np.mean([r['mse'] for r in rows if r['subset'] == 'top1000']))


def run(args):
    global STOP
    STOP = False
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested, but no GPU is connected. Select a GPU Colab runtime.')
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    data = CellData(args.cache, args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    code = provenance()
    config = dict(profile=args.profile, prior=args.prior, aggregation=args.aggregation, split=args.split,
                  seed=args.seed, hidden=args.hidden, epochs=args.epochs, batch_size=args.batch_size,
                  amp=args.amp, lr=args.lr, lr_step=args.lr_step, lr_gamma=args.lr_gamma,
                  weight_decay=5e-4, direction_lambda=.1, clip_value=1.,
                  graph_sha256=sha256(args.cache / 'graphs.npz'),
                  cell_source_sha256=data.meta['raw_sha256'],
                  profiles_sha256=sha256(args.cache / 'profiles.npz'),
                  splits_sha256=sha256(args.cache / 'splits.json'),
                  upstream_commit=code['upstream_commit'],
                  implementation_hashes={k: code['source_hashes'][k] for k in ('model.py', 'train.py', 'data.py')})
    if args.prior != 'native':
        config['prior_sha256'] = sha256(args.cache / f'{args.prior}.npz')
    latest = args.output / 'latest.pt'
    best_path = args.output / 'best.pt'
    if latest.exists():
        saved = load_checkpoint(latest)
        if saved['config'] != config:
            raise ValueError('Resume configuration/source mismatch. Use a new output directory; no overwrite performed.')
        if saved.get('complete') and (args.output / 'result.json').exists():
            print('Already completed:', args.output, flush=True)
            return 0
    else:
        saved = None
    model = PerturbationModel(data.model_args(args.hidden, args.device), data.prior(args.prior), args.aggregation).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    amp_enabled = args.device.startswith('cuda') and args.amp != 'off'
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and args.amp == 'fp16')
    amp_dtype = torch.float16 if args.amp == 'fp16' else torch.bfloat16
    epoch, offset, updates, skipped_updates = 0, 0, 0, 0
    best = math.inf
    best_epoch = None
    history = []
    loss_sum, loss_batches = 0., 0
    elapsed_prior = 0.
    if saved:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        epoch, offset, updates = saved['epoch'], saved['offset'], saved['updates']
        skipped_updates = saved['skipped_updates']
        best, best_epoch, history = saved['best'], saved['best_epoch'], saved['history']
        loss_sum, loss_batches = saved['loss_sum'], saved['loss_batches']
        elapsed_prior = saved['elapsed_seconds']
        restore_rng(saved['rng'])
        del saved
        print(f'Resuming epoch {epoch + 1}, cell offset {offset}, optimizer updates {updates}', flush=True)
    masks = data.masks(args.device)
    control = torch.from_numpy(data.control).to(args.device)
    mapping = data.control_mapping(args.seed)
    started = time.monotonic()
    last_save = started
    metadata = dict(config=config, provenance=code, device=args.device,
                    gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    parameters=sum(p.numel() for p in model.parameters()),
                    training_cells=len(data.train_rows), readout_genes=data.cells.shape[1],
                    inference='batch size 1; exact control-mean evaluation of additive input path',
                    checkpoint_selection='Mean per-validation-pair MSE on control-selected top1000 genes',
                    differences_from_published_protocol=['Processed 124-pair benchmark release',
                        'Fixed pair partitions and primary top1000 validation metric',
                        'Batch-one mean inference, deterministic control pairing per initialization',
                        'Mixed precision when amp is enabled'] +
                        (['Extended preset: larger hidden dimension and slower learning-rate decay'] if args.profile == 'extended' else []))
    atomic_json(metadata, args.output / 'run.json')
    freeze = '\n'.join(sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()))
    (args.output / 'environment.txt').write_text(freeze)

    def checkpoint(complete=False):
        nonlocal last_save
        atomic_save(dict(config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
                         epoch=epoch, offset=offset, updates=updates, best=best, best_epoch=best_epoch,
                         skipped_updates=skipped_updates,
                         loss_sum=loss_sum, loss_batches=loss_batches, history=history,
                         complete=complete, elapsed_seconds=elapsed_prior + time.monotonic() - started), latest)
        atomic_json(dict(state='complete' if complete else 'checkpointed', epoch=epoch, next_cell_offset=offset,
                         updates=updates, best_validation_mse=best if math.isfinite(best) else None,
                         elapsed_seconds=elapsed_prior + time.monotonic() - started), args.output / 'status.json')
        last_save = time.monotonic()

    def validate(partial=False):
        nonlocal best, best_epoch
        validation = predict_means(model, data, data.split['validation'], args.device)
        vi = [data.index[c] for c in data.split['validation']]
        val = float(np.mean((validation[:, data.top].astype('float64') - data.means[vi][:, data.top]) ** 2))
        if not math.isfinite(val):
            raise FloatingPointError('Nonfinite validation error')
        history.append(dict(epoch=epoch + 1, partial_epoch=partial,
                            training_gears_loss=loss_sum / max(loss_batches, 1), validation_mse=val,
                            lr=optimizer.param_groups[0]['lr'], updates=updates,
                            skipped_amp_updates=skipped_updates,
                            elapsed_seconds=elapsed_prior + time.monotonic() - started))
        pd.DataFrame(history).to_csv(args.output / 'history.csv', index=False)
        print(history[-1], flush=True)
        if val < best:
            best, best_epoch = val, epoch + 1
            atomic_save(dict(model=model.state_dict(), config=config, validation_mse=val,
                             epoch=best_epoch, partial_epoch=partial, updates=updates), best_path)

    def finish(complete):
        best_state = load_checkpoint(best_path)
        model.load_state_dict(best_state['model'])
        test_mse = score_test(model, data, config, args.output, args.device)
        atomic_json(dict(best_epoch=best_epoch, selected_partial_epoch=best_state.get('partial_epoch', False),
                         validation_mse=best, test_mse=test_mse, epochs_completed=epoch,
                         optimizer_updates=updates, state='complete' if complete else 'time_limited',
                         skipped_amp_updates=skipped_updates,
                         elapsed_seconds=elapsed_prior + time.monotonic() - started,
                         peak_gpu_allocated_gb=torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith('cuda') else None),
                    args.output / 'result.json')
        print('Finished:', args.output, 'state:', 'complete' if complete else 'time_limited',
              'test MSE:', test_mse, flush=True)
        return 0

    # Save before training, so even a first-batch OOM leaves a reviewable config.
    checkpoint()
    while epoch < args.epochs:
        model.train()
        order = np.random.default_rng(args.seed + epoch * 104729).permutation(data.train_rows)
        usable = len(order) // args.batch_size * args.batch_size  # upstream drop_last=True
        if usable == 0:
            raise ValueError('Training set smaller than one physical batch')
        while offset < usable:
            deadline_near = args.deadline_epoch is not None and time.time() >= args.deadline_epoch - 125
            if STOP or deadline_near or time.monotonic() - started >= args.max_hours * 3600:
                if args.finalize_on_budget and updates > 0:
                    if loss_batches:
                        validate(partial=True)
                    checkpoint()
                    return finish(complete=False)
                checkpoint()
                print('Session budget/stop reached. Rerun the same command to resume.', flush=True)
                return 75
            rows = order[offset:offset + args.batch_size]
            x, y, names, perts = data.inputs(rows, mapping)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                pred = model(make_batch(x, perts, args.device))
                loss = gears_loss(pred, torch.from_numpy(y).to(args.device), names, control, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss; latest safe checkpoint retained')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            offset += len(rows)
            if scaler.get_scale() >= old_scale:
                updates += 1
            else:
                skipped_updates += 1
            loss_sum += loss.item()
            loss_batches += 1
            if (updates + skipped_updates) % 100 == 0:
                print(f'epoch={epoch+1} cells={offset}/{usable} loss={loss.item():.5f} '
                      f'lr={optimizer.param_groups[0]["lr"]:.3g}', flush=True)
            if time.monotonic() - last_save >= args.checkpoint_seconds:
                checkpoint()
        # No test predictions are computed during fitting or checkpoint selection.
        validate()
        scheduler.step()
        epoch += 1
        offset, loss_sum, loss_batches = 0, 0., 0
        checkpoint()
    checkpoint(complete=True)
    return finish(complete=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT / 'gpu/cache')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=['bounded', 'upstream', 'extended'], default='bounded')
    p.add_argument('--prior', choices=['native', 'genept', 'depmap'], default='native')
    p.add_argument('--aggregation', choices=['sum', 'attention'], default='sum')
    p.add_argument('--split', type=int, default=101)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--amp', choices=['off', 'fp16', 'bf16'], default='off')
    p.add_argument('--device', default='cuda')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-step', type=int, default=1)
    p.add_argument('--lr-gamma', type=float, default=.5)
    p.add_argument('--max-hours', type=float, default=3)
    p.add_argument('--deadline-epoch', type=float)
    p.add_argument('--finalize-on-budget', action='store_true')
    p.add_argument('--checkpoint-seconds', type=float, default=300)
    p.add_argument('--threads', type=int, default=2)
    return p


def main():
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    args = parser().parse_args()
    if args.batch_size < 2 or args.epochs < 1 or args.hidden % 4 or not 0 < args.max_hours <= 3:
        raise ValueError('Use batch>=2, epochs>=1, hidden divisible by4, and at most three hours')
    args.deadline_epoch = min(args.deadline_epoch or float('inf'), time.time() + args.max_hours * 3600)
    if args.deadline_epoch <= time.time():
        print('Deadline expired; no training started.', flush=True)
        return 75
    # Independent of Python training-loop progress or checkpoint I/O.
    watchdog = threading.Timer(max(0, args.deadline_epoch - time.time()),
                               lambda: os.kill(os.getpid(), signal.SIGKILL))
    watchdog.daemon = True
    watchdog.start()
    try:
        return run(args)
    except torch.cuda.OutOfMemoryError:
        print('GPU memory exhausted. Keep the latest checkpoint. For a NEW matched experiment suite, '
              'lower the SAME physical batch size for every configuration or enable AMP; '
              'do not silently change batch size in an existing run.', file=sys.stderr)
        return 1
    finally:
        watchdog.cancel()


if __name__ == '__main__':
    sys.exit(main())
