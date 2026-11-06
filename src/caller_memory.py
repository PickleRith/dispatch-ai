"""Memory of previous callers for Dispatch AI, keyed by caller ID.

Simultaneous calls can run in separate processes, and a restart would lose a
Python variable, so callers live in a JSON file. Every read-modify-write holds a
lock file, and writes go to a temp file that replaces the data file in one step.
Standard library only; works on Windows, macOS and Linux.

Only what is needed to welcome a caller back is stored: names, the service
address, the last issue and urgency, a call count and the last call time.
Set CALLER_MEMORY=off to turn memory off without code changes.
"""

import json
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

LOCK_RETRY_SECONDS = 0.02
LOCK_TIMEOUT_SECONDS = 5.0
# A lock older than this was left by a crashed call and is removed.
LOCK_STALE_SECONDS = 10.0


class CallerMemoryBusyError(RuntimeError):
    """The callers lock could not be taken in time."""


def callers_path() -> Path:
    return Path(os.environ.get("DISPATCH_CALLERS_FILE", "callers.json"))


def enabled() -> bool:
    return os.environ.get("CALLER_MEMORY", "").strip().lower() != "off"


def normalize(phone: str | None) -> str | None:
    """The 10-digit key for a phone number, or None if it is not usable."""
    digits = "".join(c for c in phone or "" if c.isdigit())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


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
                raise CallerMemoryBusyError(
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


def _read(path: Path) -> dict:
    """Load all callers. Call only under the lock."""
    if not path.exists():
        return {}
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


def lookup(phone: str | None) -> dict | None:
    """The saved record for this caller ID, or None."""
    key = normalize(phone)
    if key is None:
        return None
    path = callers_path()
    # Reads take the lock too: on Windows, os.replace() fails while a file is open.
    with _locked(path):
        return _read(path).get(key)


def save_booking(
    phone: str | None, name: str, address: str, issue: str | None, urgency: str | None
) -> bool:
    """Create or update the caller's record after a booking. False if no usable ID."""
    key = normalize(phone)
    if key is None:
        return False
    path = callers_path()
    with _locked(path):
        data = _read(path)
        record = data.get(key, {"call_count": 0})
        data[key] = {
            "first_name": name.split()[0] if name.strip() else "",
            "full_name": name,
            "address": address,
            "last_issue": issue,
            "last_urgency": urgency,
            "call_count": record["call_count"] + 1,
            "last_call_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _write(path, data)
    return True


def forget(phone: str | None) -> bool:
    """Delete one caller's record. False if there was none."""
    key = normalize(phone)
    if key is None:
        return False
    path = callers_path()
    with _locked(path):
        data = _read(path)
        if data.pop(key, None) is None:
            return False
        _write(path, data)
    return True


def reset_all() -> None:
    """Delete the callers file."""
    with suppress(FileNotFoundError):
        callers_path().unlink()
