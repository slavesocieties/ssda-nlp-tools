"""Keep the machine awake while the overnight run is working, and no longer.

Ronak asked for this: the runs take 20+ minutes each with no keyboard activity,
and if the machine sleeps mid-merge the whole night stalls.

Deliberately NOT a power-settings change. `powercfg` would edit the machine's
configuration persistently and leave it edited if this process died. Instead
this holds `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`, the
same per-process request a media player makes while playing. Windows releases it
automatically when the process exits, however it exits, so there is nothing to
clean up and nothing left changed.

The display is deliberately NOT held awake -- the screen can and should turn off.

It stops holding the machine awake once the same heartbeat the shutdown watchdog
reads goes stale, so "the work has stopped" releases the lock and lets the
machine sleep or be shut down normally.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from datetime import datetime, timezone

def _args():
    """Refuse option-looking arguments before the execution-state lock is taken.

    Same defect as `watchdog_shutdown.py`: argument 1 is a FILE PATH, so
    `--help` became the heartbeat path and this held the machine awake for a
    full poll instead of printing usage. `self_check.py` did that on every run.
    """
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0)
    if argv and argv[0].startswith("-"):
        raise SystemExit(
            f"{os.path.basename(sys.argv[0])}: unknown option {argv[0]!r}. "
            f"Argument 1 is the heartbeat file path, argument 2 the stale "
            f"limit in minutes. Nothing was held awake.")
    return argv


_ARGV = _args()
HEARTBEAT = _ARGV[0] if _ARGV else "watchdog.heartbeat"
STALE_MINUTES = float(_ARGV[1]) if len(_ARGV) > 1 else 45.0
LOG = "keep_awake.log"
POLL_SECONDS = 60
MAX_HOURS = 9.0

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def log(msg):
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}  {msg}"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def main():
    k32 = ctypes.windll.kernel32
    if not k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        log("SetThreadExecutionState FAILED -- the machine may still sleep")
        return 1
    log(f"holding the system awake (display NOT held); heartbeat={HEARTBEAT!r}, "
        f"releasing after {STALE_MINUTES:.0f} min idle")
    started = time.time()
    try:
        while True:
            time.sleep(POLL_SECONDS)
            if not os.path.exists(HEARTBEAT):
                log("heartbeat gone -- releasing")
                return 0
            idle = (time.time() - os.path.getmtime(HEARTBEAT)) / 60.0
            if idle >= STALE_MINUTES:
                log(f"idle {idle:.1f} min -- work has stopped, releasing so the "
                    f"machine can sleep or shut down")
                return 0
            if (time.time() - started) / 3600.0 > MAX_HOURS:
                log(f"{MAX_HOURS}h ceiling -- releasing")
                return 0
    finally:
        # Drop back to the default policy. Windows would do this at exit anyway;
        # doing it explicitly means the release is visible in the log.
        k32.SetThreadExecutionState(ES_CONTINUOUS)
        log("released; normal power policy restored")


if __name__ == "__main__":
    raise SystemExit(main())
