"""Shared technician schedule for Dispatch AI.

Simultaneous calls can run in separate processes, so a Python variable is not
shared between them. The schedule lives in a JSON file instead. Every
read-modify-write holds a lock file, and writes go to a temp file that replaces
the data file in one step, so two calls can never get the same slot and a crash
never leaves half a file. Standard library only; works on Windows, macOS and
Linux.

Emergencies are first come, first served: a reservation is never moved once it
has been promised to a caller.
"""

import json
import math
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Demo data: technicians for emergency and urgent dispatch, with travel minutes.
TECHNICIANS = {"Mike Torres": 35, "Sarah Kim": 50}
# How long a job blocks a technician after they arrive (a demo assumption).
JOB_MINUTES = 45
# Routine jobs go into scheduled slots with this technician.
ROUTINE_TECHNICIAN = "Ana Ruiz"
SLOT_DAYS = 3
SLOT_TIMES = ["9:00 AM", "1:00 PM", "4:00 PM"]

LOCK_RETRY_SECONDS = 0.02
LOCK_TIMEOUT_SECONDS = 5.0
# A lock older than this was left by a crashed call and is removed.
LOCK_STALE_SECONDS = 10.0


class ScheduleBusyError(RuntimeError):
    """The schedule lock could not be taken in time."""


def schedule_path() -> Path:
    return Path(os.environ.get("DISPATCH_SCHEDULE_FILE", "schedule.json"))


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        # Windows raises PermissionError while another process deletes the lock.
        except (FileExistsError, PermissionError):
            try:
                if time.time() - os.path.getmtime(lock) > LOCK_STALE_SECONDS:
                    os.remove(lock)
                    continue
            except (FileNotFoundError, PermissionError):
                continue
            if time.monotonic() > deadline:
                raise ScheduleBusyError(
                    f"Could not lock {path} within {LOCK_TIMEOUT_SECONDS} seconds"
                ) from None
            time.sleep(LOCK_RETRY_SECONDS)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        with suppress(FileNotFoundError):
            os.remove(lock)


def _new_schedule() -> dict:
    today = date.today()
    return {
        "technicians": {
            name: {"travel_minutes": travel, "busy_until": None, "jobs": []}
            for name, travel in TECHNICIANS.items()
        },
        "slots": {
            (today + timedelta(days=n)).isoformat(): list(SLOT_TIMES)
            for n in range(1, SLOT_DAYS + 1)
        },
    }


def _read(path: Path) -> dict:
    """Load the schedule, creating it on first use. Call only under the lock."""
    if not path.exists():
        data = _new_schedule()
        _write(path, data)
        return data
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    """Write atomically: temp file in the same directory, then os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.remove(tmp)
        raise


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _busy_until(tech: dict) -> datetime | None:
    return datetime.fromisoformat(tech["busy_until"]) if tech["busy_until"] else None


def _round_up_5(minutes: float) -> int:
    return max(5, math.ceil(round(minutes, 6) / 5) * 5)


def _pick(data: dict, now: datetime) -> tuple[str, datetime, bool]:
    """The technician who can arrive first, their arrival, and whether all are busy."""
    best: tuple[str, datetime] | None = None
    all_busy = True
    for name, tech in data["technicians"].items():
        busy = _busy_until(tech)
        if busy is None or busy <= now:
            # Expired reservations free themselves.
            all_busy, start = False, now
        else:
            start = busy
        arrival = start + timedelta(minutes=tech["travel_minutes"])
        if best is None or arrival < best[1]:
            best = (name, arrival)
    assert best is not None
    return best[0], best[1], all_busy


def estimate_emergency(now: datetime | None = None) -> dict:
    """Who would be sent to an emergency right now, without reserving anyone."""
    now = _now(now)
    path = schedule_path()
    with _locked(path):
        data = _read(path)
    name, arrival, all_busy = _pick(data, now)
    return {
        "technician": name,
        "wait_minutes": _round_up_5((arrival - now).total_seconds() / 60),
        "all_busy": all_busy,
    }


def reserve_emergency(caller_label: str, now: datetime | None = None) -> dict:
    """Reserve the technician who can arrive first. Never double-books."""
    now = _now(now)
    path = schedule_path()
    with _locked(path):
        data = _read(path)
        name, arrival, _ = _pick(data, now)
        tech = data["technicians"][name]
        # Jobs that ended already are history; drop them to keep the file small.
        tech["jobs"] = [
            job for job in tech["jobs"] if datetime.fromisoformat(job["end"]) > now
        ]
        jobs_ahead = len(tech["jobs"])
        end = arrival + timedelta(minutes=JOB_MINUTES)
        tech["jobs"].append(
            {
                "caller": caller_label,
                "arrival": arrival.isoformat(),
                "end": end.isoformat(),
            }
        )
        tech["busy_until"] = end.isoformat()
        _write(path, data)
    return {
        "technician": name,
        "wait_minutes": _round_up_5((arrival - now).total_seconds() / 60),
        "jobs_ahead": jobs_ahead,
    }


def all_slots() -> dict[str, list[str]]:
    """Open routine slots by ISO date, in date order."""
    path = schedule_path()
    with _locked(path):
        slots = _read(path)["slots"]
    return dict(sorted(slots.items()))


def list_slots(day: str) -> list[str]:
    """Open routine slots on one ISO date."""
    return all_slots().get(day, [])


def reserve_slot(day: str, time_label: str) -> bool:
    """Take a routine slot. False if it is not open (or was just taken)."""
    path = schedule_path()
    with _locked(path):
        data = _read(path)
        open_times = data["slots"].get(day, [])
        if time_label not in open_times:
            return False
        open_times.remove(time_label)
        _write(path, data)
    return True


def reset_schedule() -> None:
    """Delete the schedule file, so the next call starts fresh."""
    with suppress(FileNotFoundError):
        schedule_path().unlink()
