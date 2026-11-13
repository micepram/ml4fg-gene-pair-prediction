"""Two matched full-cell fits under one immutable three-hour allowance."""
import argparse
import fcntl
import hashlib
import json
import sys
import time
import tempfile
from pathlib import Path
from .common import ROOT, atomic_json, sha256
from .budget import open_budget, remaining, run_capped, matched_epochs

SPLITS = [101, 202, 303, 404, 505]
SEEDS = [17, 29]
CONFIGURATIONS = [('native', 'sum'), ('depmap', 'attention')]


def jobs(profile='bounded', splits=(101,), seeds=(17,)):
    if profile != 'bounded' or list(splits) != [101] or list(seeds) != [17]:
        raise ValueError('The three-hour stage is fixed to split101 and seed17')
    for prior, aggregation in CONFIGURATIONS:
        yield profile, 101, 17, prior, aggregation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT / 'gpu/cache')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=['bounded'], default='bounded')
    p.add_argument('--max-hours', type=float, default=3.)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--amp', choices=['off', 'fp16'], default='fp16')
    args = p.parse_args()
    if not 0 < args.max_hours <= 3:
        p.error('The combined allowance must be at most three hours')
    args.output.mkdir(parents=True, exist_ok=True)
    lock_id = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()
    lock = open(Path(tempfile.gettempdir()) / f'ml4fg_{lock_id}.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This experiment is already running; a second training process was not launched')
    budget_path = args.output / 'budget.json'
    if budget_path.exists() and remaining(open_budget(budget_path, args.max_hours * 3600)) <= 60:
        print('The saved allowance has expired. No GPU work will be launched.', flush=True)
        return
    # Keep a full minute unused as a margin below the user\'s cap.
    plan_path = args.output / 'plan_bounded.json'
    compatibility = dict(batch_size=args.batch_size, amp=args.amp,
                         graph_sha256=sha256(args.cache / 'graphs.npz'),
                         implementation_hashes={name: sha256(ROOT / 'gpu' / name)
                                                for name in ['train.py', 'model.py', 'data.py', 'benchmark.py']})
    ready = json.loads((args.cache / 'READY.json').read_text())
    if not ready.get('ready') or ready['graph_sha256'] != compatibility['graph_sha256']:
        raise ValueError('Preparation incomplete or graph cache changed; no training allowance started')
    budget = open_budget(budget_path, args.max_hours * 3600)
    hard_deadline = budget['hard_deadline_epoch'] - 60
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan['compatibility'] != compatibility:
            raise ValueError('Settings or implementation changed; this bounded experiment cannot silently resume')
    else:
        benchmark = args.output / 'benchmark.json'
        command = [sys.executable, '-u', '-m', 'gpu.benchmark', '--cache', str(args.cache),
                   '--output', str(benchmark), '--batch-size', str(args.batch_size), '--amp', args.amp]
        code, killed = run_capped(command, min(hard_deadline, time.time() + 300), grace=15, cwd=ROOT)
        if code or killed:
            raise RuntimeError('GPU checks/timing did not finish safely; full training was not started')
        timings = json.loads(benchmark.read_text())
        epochs = matched_epochs(hard_deadline - time.time(), timings['steps_per_epoch'],
                                [r['seconds_per_step'] for r in timings['models']])
        if epochs < 1:
            print('Timing shows that one matched epoch per model will not fit. No full fits launched.', flush=True)
            return
        plan = dict(compatibility=compatibility, jobs=list(jobs()), epochs=epochs, hidden=64,
                    split=101, seed=17, training='Individual cells, official GO graph, upstream loss',
                    selection='Timing only; no validation/test performance used to choose epoch count',
                    hard_deadline_epoch=budget['hard_deadline_epoch'])
        atomic_json(plan, plan_path)
    print(f'Two matched models, {plan["epochs"]} epochs each, 64 hidden dimensions. '
          f'Total allowance remaining: {remaining(budget)/60:.1f} minutes.', flush=True)
    for index, (prior, aggregation) in enumerate(CONFIGURATIONS):
        out = args.output / f'bounded_{prior}_{aggregation}_split101_seed17'
        if (out / 'result.json').exists():
            continue
        available = hard_deadline - time.time()
        if available <= 120:
            break
        fit_deadline = time.time() + available / (len(CONFIGURATIONS) - index)
        command = [sys.executable, '-u', '-m', 'gpu.train', '--cache', str(args.cache), '--output', str(out),
                   '--profile', 'bounded', '--prior', prior, '--aggregation', aggregation,
                   '--split', '101', '--seed', '17', '--epochs', str(plan['epochs']), '--hidden', '64',
                   '--batch-size', str(args.batch_size), '--amp', args.amp,
                   '--max-hours', str((fit_deadline - time.time()) / 3600),
                   '--deadline-epoch', str(fit_deadline - 5), '--finalize-on-budget',
                   '--checkpoint-seconds', '120']
        print('Starting', out.name, flush=True)
        code, killed = run_capped(command, fit_deadline, grace=120, cwd=ROOT)
        if killed:
            print('Hard stop enforced. Last safe checkpoint retained; this fit is incomplete.', flush=True)
        elif code not in (0, 75):
            raise RuntimeError(f'{out.name} failed (exit {code}); stopping the bounded experiment')
    if hard_deadline - time.time() > 10:
        run_capped([sys.executable, '-u', '-m', 'gpu.summarize', '--runs', str(args.output)],
                   hard_deadline, grace=5, cwd=ROOT)
    atomic_json(dict(state='stopped', stopped_epoch=time.time(),
                     elapsed_wall_seconds=time.time() - budget['started_epoch'],
                     hard_deadline_epoch=budget['hard_deadline_epoch'],
                     scope='Two-model preliminary comparison; no repeated-split conclusion'), args.output / 'budget_status.json')
    print('Bounded experiment stopped. Rerunning does not reset its allowance.', flush=True)


if __name__ == '__main__':
    main()
