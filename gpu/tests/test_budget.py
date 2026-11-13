import contextlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from gpu.budget import MAX_SECONDS, matched_epochs, open_budget, remaining, run_capped
from gpu.console import run_command


class BudgetTests(unittest.TestCase):
    def test_reopening_does_not_extend_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'budget.json'
            first = open_budget(path, now=1000)
            again = open_budget(path, now=1000 + MAX_SECONDS + 100)
            self.assertEqual(first, again)
            self.assertEqual(remaining(again, now=1000 + MAX_SECONDS + 100), 0)
            self.assertEqual(first['hard_deadline_epoch'] - first['started_epoch'], 10800)
            with self.assertRaises(ValueError):
                open_budget(path, seconds=4 * 3600)
            with self.assertRaises(ValueError):
                open_budget(path, seconds=2 * 3600)

    def test_expired_budget_launches_no_process(self):
        with patch('gpu.budget.subprocess.Popen') as popen:
            self.assertEqual(run_capped([sys.executable], deadline=time.time() - 1), (75, False))
            popen.assert_not_called()

    def test_hard_stop_kills_uncooperative_process(self):
        script = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)'
        start = time.monotonic()
        code, killed = run_capped([sys.executable, '-c', script], time.time() + .6, grace=.2)
        self.assertTrue(killed)
        self.assertEqual(code, -signal.SIGKILL)
        self.assertLess(time.monotonic() - start, 2)

    def test_matched_epochs_use_only_timing_and_safety_margin(self):
        self.assertEqual(matched_epochs(10800, 2000, [.1, .15]), 15)
        self.assertEqual(matched_epochs(100, 2000, [.1, .15]), 0)
        self.assertEqual(matched_epochs(10800, 10, [.1, .15]), 20)

    def test_notebook_prints_child_output_and_errors(self):
        output = io.StringIO()
        script = 'import sys; print("preparation step"); print("specific error details",file=sys.stderr); sys.exit(3)'
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, 'code 3'):
                run_command([sys.executable, '-u', '-c', script])
        self.assertIn('preparation step', output.getvalue())
        self.assertIn('specific error details', output.getvalue())


if __name__ == '__main__':
    unittest.main()
