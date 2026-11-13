"""Stream child stdout AND stderr into notebook-visible Python output."""
import os
import signal
import subprocess


def run_command(command, cwd=None):
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        raise
    finally:
        process.stdout.close()
    if code:
        raise RuntimeError(f'Command exited with code {code}; see the printed output above')
    return code
