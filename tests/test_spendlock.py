"""The spend ledgers had a time-of-check/time-of-use race.

Both do read -> check-against-cap -> write with no lock, so two approved jobs
launched in two terminals each read the same balance, each pass the cap check,
and the second write discards the first's reservation. Measured before the fix:
$6 of spend authorised against a $4 cap, with the ledger reporting $3.

On Windows it was masked by accident -- Path.replace raises WinError 32 when the
target is open, so one racer crashed instead of double-spending. That is
filesystem luck, it surfaces as an unhandled traceback, and on POSIX os.replace
is atomic and would not raise at all.
"""
import json
import os
import threading
from pathlib import Path

import pytest

from ssda_nlp_tools.spendlock import LedgerBusy, exclusive


def test_lock_is_mutually_exclusive():
    """os.open(O_CREAT|O_EXCL) is atomic on both Windows and POSIX, which is why
    the lock is a file rather than anything cleverer."""
    import tempfile
    led = Path(tempfile.mkdtemp()) / "l.json"
    inside, overlaps, errors = [], [], []
    def worker():
        try:
            with exclusive(led, timeout=10):
                inside.append(1)
                if len(inside) > 1:
                    overlaps.append(1)
                inside.pop()
        except Exception as exc:
            errors.append(exc)
    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert overlaps == []
    assert errors == []


def test_lock_is_released_even_when_the_body_raises():
    """A failed submission must not strand the ledger. If it did, every later
    run would block for the timeout and then refuse, which for a spend guard
    looks identical to being over cap."""
    import tempfile
    led = Path(tempfile.mkdtemp()) / "l.json"
    with pytest.raises(ValueError):
        with exclusive(led, timeout=5):
            raise ValueError("submission failed")
    with exclusive(led, timeout=1):          # acquires immediately
        pass
    assert not Path(str(led) + ".lock").exists()


def test_a_held_lock_fails_closed_with_a_usable_message():
    """FAILS CLOSED on purpose. A stale lock costs a person ten seconds; a
    broken cap costs money. The message has to say how to clear it AND warn that
    a stale lock means a run died mid-flight, so its charge may be real and
    unrecorded."""
    import tempfile
    led = Path(tempfile.mkdtemp()) / "l.json"
    lock = Path(str(led) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    with pytest.raises(LedgerBusy) as exc:
        with exclusive(led, timeout=0.2, poll=0.01):
            pass
    msg = str(exc.value)
    assert str(lock) in msg and "unrecorded" in msg
    lock.unlink()


def test_concurrent_reservations_cannot_exceed_the_cap():
    """The regression this exists to prevent: four racers, a $4 cap, $3 each."""
    import importlib.util, tempfile
    spec = importlib.util.spec_from_file_location("rtb", "run_transcription_bakeoff.py")
    rtb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rtb)
    led = Path(tempfile.mkdtemp()) / "l.json"

    class A:
        reservation_usd, max_usd, ledger, model = 3.0, 4.0, str(led), "m"

    bar = threading.Barrier(4)
    ok = []
    def racer(n):
        bar.wait()
        try:
            rtb._reserve_probe(A, f"p{n}"); ok.append(n)
        except ValueError:
            pass
    ts = [threading.Thread(target=racer, args=(i,)) for i in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    final = json.loads(led.read_text(encoding="utf-8"))
    assert len(ok) == 1                                   # only one fits
    assert final["reserved_usd"] == pytest.approx(3.0)    # ledger matches reality
    assert len(final["reservations"]) == 1
