"""Run the prespecified configurations sequentially within one Colab session."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, atomic_json

SPLITS = [101, 202, 303, 404, 505]
SEEDS = [17, 29]
# Begin with the comparator and prespecified primary extension, then all controls.
CONFIGURATIONS = [('native', 'sum'), ('depmap', 'attention'), ('genept', 'sum'),
                  ('genept', 'attention'), ('depmap', 'sum'), ('native', 'attention')]


def jobs(profile, splits=SPLITS, seeds=SEEDS):
    for seed in seeds:
        for split in splits:
            if profile == 'extended':
                yield 'upstream', split, seed, 'native', 'sum'
            for prior, aggregation in CONFIGURATIONS:
                yield profile, split, seed, prior, aggregation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=ROOT / 'gpu/cache')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=['upstream', 'extended'], default='upstream')
    p.add_argument('--max-hours', type=float, default=8)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--amp', choices=['off', 'fp16', 'bf16'], default='off')
    p.add_argument('--splits', type=int, nargs='+', default=SPLITS)
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = p.parse_args()
    if not set(args.splits) <= set(SPLITS) or not set(args.seeds) <= set(SEEDS):
        raise ValueError('Use the prespecified splits and initialization seeds')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = list(jobs(args.profile, args.splits, args.seeds))
    atomic_json(dict(profile=args.profile, jobs=plan, total=len(plan),
                     interpretation='All configurations share batch size, AMP policy, split, seed, and training budget within each profile'),
                args.output / f'plan_{args.profile}.json')
    started = time.monotonic()
    for profile, split, seed, prior, aggregation in plan:
        remaining = args.max_hours - (time.monotonic() - started) / 3600
        if remaining <= 0:
            break
        out = args.output / f'{profile}_{prior}_{aggregation}_split{split}_seed{seed}'
        cmd = [sys.executable, '-u', '-m', 'gpu.train', '--cache', str(args.cache), '--output', str(out),
               '--profile', profile, '--prior', prior, '--aggregation', aggregation,
               '--split', str(split), '--seed', str(seed), '--max-hours', str(remaining),
               '--batch-size', str(args.batch_size), '--amp', args.amp]
        if profile == 'extended':
            cmd += ['--hidden', '128', '--epochs', '50', '--lr-step', '10']
        print('Running', out.name, flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode == 75:
            print('Session paused at a saved checkpoint. Rerun this cell in a GPU session.', flush=True)
            break
        if result.returncode:
            raise RuntimeError(f'{out.name} failed (exit {result.returncode}); inspect the error before resuming')
    subprocess.run([sys.executable, '-m', 'gpu.summarize', '--runs', str(args.output)], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
