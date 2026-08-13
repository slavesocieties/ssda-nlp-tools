"""Shut the machine down once the assistant has clearly stopped working.

Ronak asked for this on 2026-08-10: he is asleep, the session may end at any
point (most likely by exhausting the token budget), and he does not want the
machine running all night afterwards.

There is no signal for "the assistant ran out of budget", so this infers it from
silence. The assistant touches HEARTBEAT after each milestone; if the file stops
being updated for STALE_MINUTES, the work has stopped and the machine goes down.

THREE SAFETY PROPERTIES, because this powers off someone's computer:

  1. A GENEROUS TIMEOUT. A full merge run takes ~22 minutes with no output, so a
     short timeout would kill the machine mid-run. 45 minutes is well clear of
     the longest legitimate silence.

  2. A 60-SECOND ABORT WINDOW. The actual shutdown is `shutdown /s /t 60`, so
     anyone at the keyboard has a minute to run `shutdown /a` and cancel.

  3. TRIVIALLY CANCELLABLE. Delete the heartbeat file's parent flag
     (`watchdog.disable`), or kill this process, and nothing happens. Both are
     checked every poll.

It writes what it is doing to watchdog.log, so the decision is auditable after
the fact rather than mysterious.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone

def _args():
    """Refuse option-looking arguments before anything is armed.

    Argument 1 is a FILE PATH, so `--help` used to become the heartbeat path
    and start a real shutdown timer -- which is exactly what `self_check.py`'s
    "--help is unconditionally safe" sweep did to every script in the repo,
    twice a run. A watchdog that powers off the machine must not be startable
    by a flag typed in the hope of reading its usage.
    """
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0)
    if argv and argv[0].startswith("-"):
        raise SystemExit(
            f"{os.path.basename(sys.argv[0])}: unknown option {argv[0]!r}. "
            f"Argument 1 is the heartbeat file path, argument 2 the stale "
            f"limit in minutes. Nothing was armed.")
    return argv


_ARGV = _args()
HEARTBEAT = _ARGV[0] if _ARGV else "watchdog.heartbeat"
STALE_MINUTES = float(_ARGV[1]) if len(_ARGV) > 1 else 45.0
LOG = os.path.join(os.path.dirname(os.path.abspath(HEARTBEAT)) or ".",
                   "watchdog.log")
DISABLE = os.path.join(os.path.dirname(os.path.abspath(HEARTBEAT)) or ".",
                       "watchdog.disable")
POLL_SECONDS = 60
MAX_HOURS = 9.0          # never outlive the night, whatever happens


def log(msg):
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}  {msg}"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def main():
    started = time.time()
    log(f"watchdog up: heartbeat={HEARTBEAT!r} stale_after={STALE_MINUTES:.0f}min "
        f"poll={POLL_SECONDS}s")
    log(f"to cancel: delete this process, or create {DISABLE!r}")
    while True:
        time.sleep(POLL_SECONDS)

        if os.path.exists(DISABLE):
            log("watchdog.disable present -- standing down, no shutdown")
            return 0

        if (time.time() - started) / 3600.0 > MAX_HOURS:
            log(f"reached the {MAX_HOURS}h ceiling -- standing down rather than "
                f"shutting down on a stale assumption")
            return 0

        if not os.path.exists(HEARTBEAT):
            log("heartbeat file is gone -- treating as an explicit cancel")
            return 0

        idle = (time.time() - os.path.getmtime(HEARTBEAT)) / 60.0
        if idle < STALE_MINUTES:
            continue

        log(f"heartbeat stale for {idle:.1f} min (limit {STALE_MINUTES:.0f}) -- "
            f"the assistant has stopped. Shutting down in 60s.")
        log("anyone at the keyboard can cancel with:  shutdown /a")
        try:
            subprocess.run(
                ["shutdown", "/s", "/t", "60", "/c",
                 "SSDA overnight run finished or stopped; watchdog shutdown. "
                 "Cancel with: shutdown /a"],
                check=True)
            log("shutdown scheduled")
        except Exception as exc:                     # noqa: BLE001
            log(f"shutdown command FAILED: {exc!r} -- machine left running")
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
