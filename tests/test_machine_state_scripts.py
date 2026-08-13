"""`--help` must never arm the two scripts that change machine state.

Found 2026-08-12. `self_check.py` runs every offline script with `--help` on the
stated argument that `--help` is "unconditionally SAFE to run". Two scripts have
no argparse and read `sys.argv[1]` as a FILE PATH, so the flag became the
heartbeat path and started real work:

  watchdog_shutdown.py --help  -> armed the `shutdown /s /t 60` watchdog
  keep_awake.py --help         -> took the SetThreadExecutionState lock

Both then self-cancelled after one 60s poll, but only by luck: they cancel
because no file named `--help` exists. Create one and the shutdown watchdog is
live. The visible damage was smaller and still real -- ~120s of every
`self_check` run was spent inside these two, and when a poll slipped past the
check's own 120s subprocess timeout it produced a phantom second FAIL on a run
that was otherwise clean. A verification tool that reports an extra failure
under load is worse than one that reports none.

The guard is cheap; what it protects is not. These tests fail if it is removed.
"""
import os
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["watchdog_shutdown.py", "keep_awake.py"]

# One poll is 60s. Anything at or above that means the script entered its loop
# instead of printing usage, which is the defect itself.
POLL_SECONDS = 60
BUDGET_SECONDS = 20


def _run(script, *args, timeout=POLL_SECONDS):
    started = time.time()
    proc = subprocess.run([sys.executable, "-X", "utf8", script, *args],
                          capture_output=True, text=True, cwd=ROOT,
                          timeout=timeout)
    return proc, time.time() - started


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_exits_clean(script, flag):
    proc, elapsed = _run(script, flag)
    assert proc.returncode == 0, proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout.strip(), "--help printed nothing"
    assert elapsed < BUDGET_SECONDS, (
        f"{script} {flag} took {elapsed:.0f}s -- it entered its poll loop "
        f"instead of printing usage")


@pytest.mark.parametrize("script", SCRIPTS)
def test_option_looking_argument_is_refused_not_treated_as_a_path(script):
    """The general form of the bug, not just the one flag that exposed it."""
    proc, elapsed = _run(script, "--stale-after=5")
    assert proc.returncode != 0
    assert "Traceback" not in proc.stderr
    assert "unknown option" in proc.stderr
    assert elapsed < BUDGET_SECONDS


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_does_not_write_the_log(script, tmp_path):
    """Usage output must leave no trace -- a log line means the loop was entered."""
    log = os.path.join(ROOT, "watchdog.log" if "watchdog" in script
                       else "keep_awake.log")
    before = os.path.getsize(log) if os.path.exists(log) else 0
    _run(script, "--help")
    after = os.path.getsize(log) if os.path.exists(log) else 0
    assert after == before, f"{script} --help appended to {os.path.basename(log)}"


def test_the_heartbeat_path_still_works_as_argument_one():
    """The guard must not break the real calling convention."""
    proc, _ = _run("watchdog_shutdown.py", "--help")
    assert "heartbeat" in proc.stdout.lower()
