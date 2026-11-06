"""Shared technician schedule: no double-booking, even across processes.

No LLM and no network. Each test gets its own schedule file in tmp_path through
DISPATCH_SCHEDULE_FILE. The concurrency tests use real processes started with
the spawn method, the same way LiveKit runs calls on macOS and Linux.
"""

import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

import dispatch_schedule as ds

NOW = datetime(2026, 11, 6, 22, 0, tzinfo=timezone.utc)


@pytest.fixture
def schedule_file(tmp_path, monkeypatch):
    path = tmp_path / "schedule.json"
    monkeypatch.setenv("DISPATCH_SCHEDULE_FILE", str(path))
    return path


def _reserve_worker(path, barrier, label):
    os.environ["DISPATCH_SCHEDULE_FILE"] = path
    barrier.wait()
    ds.reserve_emergency(label)


def _slot_worker(path, barrier, day, results):
    os.environ["DISPATCH_SCHEDULE_FILE"] = path
    barrier.wait()
    results.put(ds.reserve_slot(day, "9:00 AM"))


def _run_processes(targets):
    procs = [mp.get_context("spawn").Process(target=t, args=a) for t, a in targets]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    return procs


def test_two_free_technicians(schedule_file):
    first = ds.reserve_emergency("call 1", now=NOW)
    second = ds.reserve_emergency("call 2", now=NOW)
    assert (first["technician"], first["wait_minutes"]) == ("Mike Torres", 35)
    assert (second["technician"], second["wait_minutes"]) == ("Sarah Kim", 50)
    assert first["jobs_ahead"] == second["jobs_ahead"] == 0


def test_waits_grow_once_both_are_busy(schedule_file):
    results = [ds.reserve_emergency(f"call {n}", now=NOW) for n in range(1, 6)]
    waits = [r["wait_minutes"] for r in results]
    assert waits == sorted(waits)
    assert results[2]["wait_minutes"] > 50
    assert results[2]["jobs_ahead"] >= 1
    assert results[3]["jobs_ahead"] >= 1


def test_estimate_does_not_reserve(schedule_file):
    assert ds.estimate_emergency(now=NOW)["wait_minutes"] == 35
    assert ds.estimate_emergency(now=NOW)["wait_minutes"] == 35
    ds.reserve_emergency("call 1", now=NOW)
    ds.reserve_emergency("call 2", now=NOW)
    estimate = ds.estimate_emergency(now=NOW)
    assert estimate["all_busy"] is True
    assert estimate["wait_minutes"] > 50


def test_five_processes_never_double_book(schedule_file):
    barrier = mp.get_context("spawn").Barrier(5)
    _run_processes(
        [
            (_reserve_worker, (str(schedule_file), barrier, f"call {n}"))
            for n in range(5)
        ]
    )

    data = json.loads(schedule_file.read_text(encoding="utf-8"))
    jobs = [job for tech in data["technicians"].values() for job in tech["jobs"]]
    assert sorted(job["caller"] for job in jobs) == [f"call {n}" for n in range(5)]
    for tech in data["technicians"].values():
        booked = sorted(tech["jobs"], key=lambda job: job["arrival"])
        for before, after in zip(booked, booked[1:]):
            assert after["arrival"] >= before["end"]
    assert not schedule_file.with_name("schedule.json.lock").exists()


def test_two_processes_one_slot(schedule_file):
    day = next(iter(ds.all_slots()))
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(2), ctx.Queue()
    _run_processes(
        [(_slot_worker, (str(schedule_file), barrier, day, results)) for _ in range(2)]
    )
    assert sorted([results.get(timeout=5), results.get(timeout=5)]) == [False, True]
    assert "9:00 AM" not in ds.list_slots(day)


def test_slot_booked_once(schedule_file):
    day = next(iter(ds.all_slots()))
    assert ds.reserve_slot(day, "1:00 PM") is True
    assert ds.reserve_slot(day, "1:00 PM") is False
    assert ds.list_slots(day) == ["9:00 AM", "4:00 PM"]


def test_stale_lock_is_removed(schedule_file):
    lock = schedule_file.with_name("schedule.json.lock")
    lock.write_text("12345")
    old = time.time() - ds.LOCK_STALE_SECONDS - 5
    os.utime(lock, (old, old))
    assert ds.reserve_emergency("call 1", now=NOW)["technician"] == "Mike Torres"
    assert not lock.exists()


def test_fresh_lock_times_out(schedule_file, monkeypatch):
    monkeypatch.setattr(ds, "LOCK_TIMEOUT_SECONDS", 0.2)
    schedule_file.with_name("schedule.json.lock").write_text("12345")
    with pytest.raises(ds.ScheduleBusyError):
        ds.reserve_emergency("call 1", now=NOW)


def test_expired_busy_until_counts_as_free(schedule_file):
    ds.reserve_emergency("call 1", now=NOW)
    ds.reserve_emergency("call 2", now=NOW)
    later = NOW + timedelta(hours=3)
    estimate = ds.estimate_emergency(now=later)
    assert estimate["all_busy"] is False
    assert (estimate["technician"], estimate["wait_minutes"]) == ("Mike Torres", 35)
    assert ds.reserve_emergency("call 3", now=later)["jobs_ahead"] == 0


def test_reset_schedule(schedule_file):
    ds.reserve_emergency("call 1", now=NOW)
    assert schedule_file.exists()
    ds.reset_schedule()
    assert not schedule_file.exists()
