"""Persisted wall-clock deadline and a process-group hard stop."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import atomic_json

MAX_SECONDS = 3 * 60 * 60


def open_budget(path, seconds=MAX_SECONDS, now=None):
    """Reopening an existing budget never moves its deadline forward.

    Pauses also count, so reconnecting cannot grant another three hours.
    """
    if not 0 < seconds <= MAX_SECONDS:
        raise ValueError('The total allowance must be positive and at most three hours')
    path = Path(path)
    if path.exists():
        value = json.loads(path.read_text())
        if value['limit_seconds'] != seconds:
            raise ValueError('Existing training allowance cannot be changed or reset')
        if value['hard_deadline_epoch'] != value['started_epoch'] + value['limit_seconds']:
            raise ValueError('Invalid deadline in the saved allowance')
        return value
    now = time.time() if now is None else now
    value = dict(limit_seconds=seconds, started_epoch=now, hard_deadline_epoch=now + seconds,
                 scope='Benchmark, training, validation, checkpointing and test evaluation together; pauses count')
    atomic_json(value, path)
    return value


def remaining(budget, now=None):
    return max(0., budget['hard_deadline_epoch'] - (time.time() if now is None else now))


def signal_group(process, sig):
    if process.poll() is None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass


def run_capped(command, deadline, grace=90., cwd=None):
    """Request checkpointing early, then kill the process group at deadline.

    Returns (exit code, hard-killed). An expired budget launches no process.
    stdout is inherited so the notebook can stream the runner's output.
    """
    available = deadline - time.time()
    if available <= 0:
        return 75, False
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        try:
            return process.wait(timeout=max(0., available - grace)), False
        except subprocess.TimeoutExpired:
            signal_group(process, signal.SIGTERM)
        try:
            return process.wait(timeout=max(0., deadline - time.time())), False
        except subprocess.TimeoutExpired:
            signal_group(process, signal.SIGKILL)
            process.wait()
            return process.returncode, True
    except BaseException:
        signal_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=min(grace, max(0., deadline - time.time())))
        except BaseException:
            signal_group(process, signal.SIGKILL)
            process.wait()
        raise


def matched_epochs(seconds, steps_per_epoch, seconds_per_step, maximum=20):
    """Common epoch count chosen from timing only, with a 35% margin."""
    epoch_pair = steps_per_epoch * sum(seconds_per_step) * 1.35
    available = max(0., seconds - 600)  # Loading, validation, Drive I/O and scoring.
    return min(maximum, int(available // epoch_pair)) if epoch_pair > 0 else 0
