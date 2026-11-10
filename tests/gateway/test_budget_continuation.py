"""agent.auto_continue_on_budget: gateway sessions continue past the iteration budget."""

import time
from unittest.mock import AsyncMock

import pytest

from gateway import budget_continuation as bc
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals

SID = "sid-budget"


@pytest.fixture
def db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    d = goals._get_session_db()
    d.create_session(SID, source="telegram")
    yield d
    goals._DB_CACHE.clear()


ON = bc.BudgetContinuationConfig(enabled=True, max_continuations=2)


def _decide(db, cfg=ON, **kw):
    args = dict(real_user_turn=True, continuation_turn=False, failed=False, goal_active=False,
                approval_pending=False)
    args.update(kw)
    return bc.decide(cfg, db=db, session_id=SID, **args)


def test_parse_config_defaults_off_and_accepts_shorthand():
    assert bc.parse_config(None).enabled is False
    assert bc.parse_config(True).enabled is True
    cfg = bc.parse_config({"enabled": True, "max_continuations_per_session": "3", "message": " go on "})
    assert (cfg.enabled, cfg.max_continuations, cfg.message) == (True, 3, "go on")


def test_budget_exit_queues_marked_continuation_and_caps(db):
    db.set_last_turn_exit(SID, "iteration_budget", "max_iterations_reached(150/150)")
    p1 = _decide(db)
    assert p1.startswith("[budget-continuation 1/2]") and bc.is_continuation_text(p1)
    p2 = _decide(db, real_user_turn=False, continuation_turn=True)
    assert p2.startswith("[budget-continuation 2/2]")
    assert _decide(db, real_user_turn=False, continuation_turn=True) is None  # cap


def test_real_user_message_resets_counter(db):
    db.set_last_turn_exit(SID, "iteration_budget", "x")
    _decide(db)
    _decide(db, real_user_turn=False, continuation_turn=True)
    assert bc.read_count(db, SID) == 2
    assert _decide(db).startswith("[budget-continuation 1/2]")


@pytest.mark.parametrize("reason", ["completed", "interrupted", "error", "timeout", "incomplete"])
def test_non_budget_exits_never_continue(db, reason):
    db.set_last_turn_exit(SID, reason, "")
    assert _decide(db) is None


@pytest.mark.parametrize("kw", [
    {"failed": True}, {"goal_active": True}, {"approval_pending": True},
    {"real_user_turn": False, "continuation_turn": False},  # heartbeat / loop tick / goal turn
])
def test_guards_skip(db, kw):
    db.set_last_turn_exit(SID, "iteration_budget", "x")
    assert _decide(db, **kw) is None


def test_disabled_or_stale_does_nothing(db):
    db.set_last_turn_exit(SID, "iteration_budget", "x")
    assert _decide(db, cfg=bc.BudgetContinuationConfig()) is None
    assert _decide(db, now=time.time() + 3600) is None


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")})
    runner.adapters = {}
    runner._session_key_for_source = lambda source: "agent:main:telegram:dm:1"
    runner._enqueue_synthetic_continuation = AsyncMock()
    return runner


def _event(text="do the long task", **kw):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=SessionSource(
        platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="u"), **kw)


@pytest.mark.asyncio
async def test_runner_hook_enqueues_through_goal_path(db, monkeypatch):
    monkeypatch.setattr(bc, "load_config", lambda: ON)
    db.set_last_turn_exit(SID, "iteration_budget", "max_iterations_reached(150/150)")
    runner, ev = _runner(), _event()
    entry = type("E", (), {"session_id": SID})()
    await GatewayRunner._post_turn_budget_continuation(
        runner, session_entry=entry, source=ev.source, event=ev, is_internal=False, failed=False)
    runner._enqueue_synthetic_continuation.assert_awaited_once()
    _src, prompt, producer, meta = runner._enqueue_synthetic_continuation.await_args.args
    assert prompt.startswith("[budget-continuation 1/2]") and producer == bc.PRODUCER and meta == {bc.PRODUCER: True}

    # The continuation turn itself (internal, marker text) ends on budget again -> 2/2, then cap.
    cont = _event(prompt, internal=True)
    for _ in range(2):
        await GatewayRunner._post_turn_budget_continuation(
            runner, session_entry=entry, source=cont.source, event=cont, is_internal=True, failed=False)
    assert runner._enqueue_synthetic_continuation.await_count == 2


@pytest.mark.asyncio
async def test_runner_hook_skips_completed_turn(db, monkeypatch):
    monkeypatch.setattr(bc, "load_config", lambda: ON)
    db.set_last_turn_exit(SID, "completed", "text_response(stop)")
    runner, ev = _runner(), _event()
    await GatewayRunner._post_turn_budget_continuation(
        runner, session_entry=type("E", (), {"session_id": SID})(), source=ev.source, event=ev,
        is_internal=False, failed=False)
    runner._enqueue_synthetic_continuation.assert_not_awaited()
