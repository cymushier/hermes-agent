"""chat -q --continue-on-budget N: same-session continuation past the iteration budget."""

import pytest

import cli
from hermes_cli import oneshot_budget_continuation as obc

BUDGET = {"completed": False, "failed": False, "interrupted": False,
          "turn_exit_reason": "max_iterations_reached(150/150)", "final_response": "partial"}
DONE = {"completed": True, "failed": False, "turn_exit_reason": "text_response(stop)", "final_response": "ok"}


def _drive(sequence, cap):
    """``sequence`` = results of the continuation turns, after an initial budget stop."""
    state = {"result": BUDGET}
    prompts = []
    it = iter(sequence)

    def run_turn(prompt):
        prompts.append(prompt)
        state["result"] = next(it)

    sent = obc.run_budget_continuations(run_turn, lambda: state["result"], cap)
    return sent, prompts, state["result"]


def test_continues_in_same_session_until_done():
    sent, prompts, final = _drive([BUDGET, BUDGET, DONE], cap=5)
    assert sent == 3 and final is DONE
    assert [p.split("]")[0] for p in prompts] == [
        "[budget-continuation 1/5", "[budget-continuation 2/5", "[budget-continuation 3/5"]


def test_cap_is_enforced():
    sent, _, final = _drive([BUDGET] * 10, cap=2)
    assert sent == 2 and final is BUDGET


@pytest.mark.parametrize("stop", [
    {**BUDGET, "failed": True}, {**BUDGET, "interrupted": True}, DONE,
])
def test_failure_interrupt_or_completion_stop_immediately(stop):
    sent, _, final = _drive([stop, BUDGET], cap=5)
    assert sent == 1 and final is stop


def test_zero_cap_is_noop():
    assert _drive([DONE], cap=0)[0] == 0


def test_take_continue_on_budget_flag_env_and_consumption(monkeypatch):
    monkeypatch.setenv(obc.ENV_VAR, "4")
    assert obc.take_continue_on_budget() == 4
    import os
    assert obc.ENV_VAR not in os.environ  # not inherited by tool subprocesses
    monkeypatch.setenv(obc.ENV_VAR, "9")
    assert obc.take_continue_on_budget(2) == 2
    monkeypatch.setenv(obc.ENV_VAR, "junk")
    assert obc.take_continue_on_budget() == 0
    assert obc.take_continue_on_budget(10**9) == obc.MAX_CONTINUATIONS
    assert obc.take_continue_on_budget(-3) == 0


def test_parser_accepts_flag():
    from hermes_cli._parser import build_top_level_parser
    parser, *_ = build_top_level_parser()
    args = parser.parse_args(["chat", "-q", "hi", "--continue-on-budget", "3"])
    assert args.continue_on_budget == 3


@pytest.mark.parametrize("cap,expected_calls,expected_code", [(0, 1, 3), (5, 3, 0), (1, 2, 3)])
def test_quiet_one_shot_continues_same_session(monkeypatch, capsys, cap, expected_calls, expected_code):
    """-Q end to end: budget, budget, done -> one process, one history chain, final exit code."""
    from types import SimpleNamespace
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    results = iter([BUDGET, BUDGET, DONE])
    calls = []

    def run_conversation(**kwargs):
        calls.append(kwargs)
        res = dict(next(results))
        res["messages"] = list(kwargs["conversation_history"]) + [{"role": "user", "content": kwargs["user_message"]}]
        return res

    agent = SimpleNamespace(run_conversation=run_conversation, session_id="s-1")
    fake = SimpleNamespace(agent=agent, conversation_history=[], session_id="s-1", _continue_on_budget=cap)
    with pytest.raises(SystemExit) as exc:
        cli._run_quiet_single_query(fake, "hello")
    assert exc.value.code == expected_code
    assert len(calls) == expected_calls
    for i, call in enumerate(calls[1:], start=1):
        assert call["user_message"].startswith(f"[budget-continuation {i}/{cap}]")
        assert len(call["conversation_history"]) == i  # same session context carried forward
