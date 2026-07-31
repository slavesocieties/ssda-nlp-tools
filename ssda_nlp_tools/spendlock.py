"""An exclusive lock around a spend ledger.

Both ledgers in this project do read -> check-against-cap -> write. With no
lock that is a time-of-check/time-of-use race, and it is not theoretical: two
approved jobs launched in two terminals each read the same balance, each pass
the cap check, and the second write discards the first's reservation. Measured
on a $4 cap with two $3 reservations: $6 of real spend authorised, ledger
reporting $3, one reservation vanished.

On Windows the race was masked by accident, because `Path.replace` raises
WinError 32 when the target is open, so one racer crashed instead of
double-spending. That is filesystem luck rather than a guard, it fails as an
unhandled traceback, and on POSIX `os.replace` is atomic and would not raise at
all -- both racers would succeed.

`os.open(O_CREAT | O_EXCL)` is atomic on both platforms, which is why the lock
is a file rather than anything cleverer.

FAILS CLOSED. If a previous run died holding the lock, this blocks and then
raises rather than assuming the holder is gone. For a guard whose whole job is
preventing an unintended charge, refusing to proceed is the correct failure:
a stale lock costs a person ten seconds, a broken cap costs money.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path


class LedgerBusy(RuntimeError):
    """Another process holds the spend ledger."""


@contextmanager
def exclusive(ledger_path, timeout: float = 30.0, poll: float = 0.05):
    """Hold an exclusive lock on `<ledger>.lock` for the whole read/write cycle.

    The lock must wrap BOTH the read and the write. Wrapping only the write
    would still let two processes read the same balance and both pass the cap
    check, which is the actual bug.
    """
    lock = Path(str(ledger_path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                holder = ""
                try:
                    holder = f" (held by pid {lock.read_text(encoding='utf-8').strip()})"
                except OSError:
                    pass
                raise LedgerBusy(
                    f"spend ledger {ledger_path} is locked{holder} after "
                    f"{timeout:g}s. Another approved job is probably running. "
                    f"If you are certain none is, delete {lock} and retry -- "
                    f"but check the ledger's reservations first, because a "
                    f"stale lock means a run died mid-flight and its charge may "
                    f"be real and unrecorded.") from None
            time.sleep(poll)
    try:
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        except OSError:
            pass                      # the pid is a courtesy, not the mechanism
        yield
    finally:
        try:
            os.close(fd)
        finally:
            try:
                lock.unlink()
            except OSError:
                pass                  # already gone; nothing to release
