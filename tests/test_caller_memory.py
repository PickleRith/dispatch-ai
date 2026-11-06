"""Caller memory: lookup and save by caller ID, safe across processes.

No LLM and no network. Each test gets its own callers file in tmp_path through
DISPATCH_CALLERS_FILE. The concurrency test uses real processes started with
the spawn method, the same way LiveKit runs calls on macOS and Linux.
"""

import json
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import pytest

import caller_memory as cm

# my_agent and livekit are imported inside the agent tests only. Spawned worker
# processes re-import this module, and pulling the whole LiveKit stack into ten
# processes at once made some of them crash on import.

PHONE = "+16465550142"
KEY = "6465550142"


@pytest.fixture
def callers_file(tmp_path, monkeypatch):
    path = tmp_path / "callers.json"
    monkeypatch.setenv("DISPATCH_CALLERS_FILE", str(path))
    monkeypatch.delenv("CALLER_MEMORY", raising=False)
    return path


@pytest.fixture
def agent_env(callers_file, tmp_path, monkeypatch):
    """Keep the agent's dispatches.json and schedule.json inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DISPATCH_SCHEDULE_FILE", str(tmp_path / "schedule.json"))
    return callers_file


def _fake_sip_call(monkeypatch, phone):
    from livekit import rtc

    import my_agent

    sip = SimpleNamespace(
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
        attributes={"sip.phoneNumber": phone},
    )
    ctx = SimpleNamespace(room=SimpleNamespace(remote_participants={"p": sip}))
    monkeypatch.setattr(my_agent, "get_job_context", lambda required=False: ctx)


def _no_call(monkeypatch):
    import my_agent

    monkeypatch.setattr(my_agent, "get_job_context", lambda required=False: None)


async def _emergency_call(name, address):
    import my_agent

    agent = my_agent.MyAgent()
    assessed = await agent.assess_urgency(None, "Burst pipe.", "emergency", ["water"])
    booked = await agent.book_technician(None, name=name, address=address)
    return assessed, booked


def _save_worker(path, barrier, phone, name):
    os.environ["DISPATCH_CALLERS_FILE"] = path
    barrier.wait(timeout=60)
    cm.save_booking(phone, name, "1 Main Street", "Burst pipe.", "emergency")


@pytest.mark.parametrize("raw", ["+16465550142", "646-555-0142", "16465550142"])
def test_normalize_same_key(raw):
    assert cm.normalize(raw) == KEY


@pytest.mark.parametrize("raw", ["555", "", None, "+44 20 7946 0958 12"])
def test_normalize_rejects(raw):
    assert cm.normalize(raw) is None


def test_save_and_lookup(callers_file):
    assert cm.lookup(PHONE) is None
    for count in (1, 2, 3):
        assert cm.save_booking(PHONE, "Dana Reyes", "12 Elm Street", "Leak.", "urgent")
        saved = cm.lookup("646-555-0142")
        assert saved["call_count"] == count
    assert saved["first_name"] == "Dana"
    assert saved["full_name"] == "Dana Reyes"
    assert set(saved) == {
        "first_name",
        "full_name",
        "address",
        "last_issue",
        "last_urgency",
        "call_count",
        "last_call_at",
    }


def test_new_address_updates(callers_file):
    cm.save_booking(PHONE, "Dana Reyes", "12 Elm Street", "Leak.", "urgent")
    cm.save_booking(PHONE, "Dana Reyes", "40 Oak Avenue", "Clog.", "routine")
    saved = cm.lookup(PHONE)
    assert saved["address"] == "40 Oak Avenue"
    assert saved["last_issue"] == "Clog."


def test_unusable_number_is_not_saved(callers_file):
    assert (
        cm.save_booking("555", "Dana Reyes", "12 Elm Street", "Leak.", "urgent")
        is False
    )
    assert not callers_file.exists()


def test_forget_and_reset(callers_file):
    cm.save_booking(PHONE, "Dana Reyes", "12 Elm Street", "Leak.", "urgent")
    assert cm.forget(PHONE) is True
    assert cm.lookup(PHONE) is None
    assert cm.forget(PHONE) is False
    cm.reset_all()
    assert not callers_file.exists()


def test_processes_save_at_once(callers_file):
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(10)
    targets = [
        (str(callers_file), barrier, f"21255501{n:02d}", f"Caller {n}")
        for n in range(5)
    ]
    targets += [(str(callers_file), barrier, PHONE, "Dana Reyes")] * 5
    procs = [ctx.Process(target=_save_worker, args=args) for args in targets]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=90)
        assert p.exitcode == 0

    data = json.loads(callers_file.read_text(encoding="utf-8"))
    assert set(data) == {f"21255501{n:02d}" for n in range(5)} | {KEY}
    assert data[KEY]["call_count"] == 5
    assert not callers_file.with_name("callers.json.lock").exists()


def test_stale_lock_is_removed(callers_file):
    lock = callers_file.with_name("callers.json.lock")
    lock.write_text("12345")
    old = time.time() - cm.LOCK_STALE_SECONDS - 5
    os.utime(lock, (old, old))
    assert cm.save_booking(PHONE, "Dana Reyes", "12 Elm Street", "Leak.", "urgent")
    assert not lock.exists()


def test_memory_off(agent_env, monkeypatch):
    monkeypatch.setenv("CALLER_MEMORY", "off")
    assert cm.enabled() is False


async def test_memory_off_tools_skip(agent_env, monkeypatch):
    cm.save_booking(PHONE, "Dana Reyes", "12 Elm Street", "Leak.", "urgent")
    before = agent_env.read_text(encoding="utf-8")
    monkeypatch.setenv("CALLER_MEMORY", "off")
    _fake_sip_call(monkeypatch, PHONE)
    assessed, booked = await _emergency_call("Dana Reyes", "40 Oak Avenue")
    assert "Returning caller" not in assessed
    assert booked.startswith("Booked")
    assert agent_env.read_text(encoding="utf-8") == before


async def test_tools_remember_returning_caller(agent_env, monkeypatch):
    _fake_sip_call(monkeypatch, PHONE)
    assessed, booked = await _emergency_call("Dana Reyes", "12 Elm Street")
    assert "Returning caller" not in assessed
    assert booked.startswith("Booked")
    assert cm.lookup(PHONE)["call_count"] == 1

    assessed, _ = await _emergency_call("Dana Reyes", "12 Elm Street")
    assert "Returning caller: first name Dana" in assessed
    assert "Do not say the saved address or full name" in assessed
    assert "12 Elm Street" in assessed
    assert cm.lookup(PHONE)["call_count"] == 2


async def test_gas_keeps_safety_first(agent_env, monkeypatch):
    cm.save_booking(PHONE, "Dana Reyes", "12 Elm Street", "Leak.", "urgent")
    import my_agent

    _fake_sip_call(monkeypatch, PHONE)
    agent = my_agent.MyAgent()
    assessed = await agent.assess_urgency(None, "Gas smell.", "emergency", ["gas"])
    assert assessed.index("Leave the building") < assessed.index("Returning caller")


async def test_console_no_lookup_or_save(agent_env, monkeypatch):
    import my_agent

    cm.save_booking(PHONE, "Dana Reyes", "12 Elm Street", "Leak.", "urgent")
    before = agent_env.read_text(encoding="utf-8")
    _no_call(monkeypatch)
    monkeypatch.setattr(cm, "lookup", lambda phone: pytest.fail("looked up"))
    agent = my_agent.MyAgent()
    assessed = await agent.assess_urgency(None, "Burst pipe.", "emergency", ["water"])
    booked = await agent.book_technician(
        None, name="Dana Reyes", address="12 Elm Street", phone="6465550142"
    )
    assert "Returning caller" not in assessed
    assert booked.startswith("Booked")
    assert agent_env.read_text(encoding="utf-8") == before
