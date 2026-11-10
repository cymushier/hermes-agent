"""Persisted turn-exit category and the one-shot budget exit code."""

import pytest

from agent.turn_exit_status import (
    COMPLETED, ERROR, INCOMPLETE, INTERRUPTED, ITERATION_BUDGET, TIMEOUT,
    classify_turn_exit, is_budget_exhausted,
)
from hermes_cli.turn_exit import BUDGET_EXHAUSTED_EXIT_CODE, turn_exit_code


def _budget():
    return {"completed": False, "failed": False, "interrupted": False,
            "turn_exit_reason": "max_iterations_reached(150/150)", "final_response": "summary"}


@pytest.mark.parametrize("result,expected", [
    ({"completed": True, "turn_exit_reason": "text_response(stop)"}, COMPLETED),
    (_budget(), ITERATION_BUDGET),
    ({**_budget(), "final_response": ""}, ITERATION_BUDGET),
    ({**_budget(), "interrupted": True}, INTERRUPTED),
    ({**_budget(), "failed": True}, ERROR),
    ({"failed": True, "failure_reason": "timeout", "turn_exit_reason": "x"}, TIMEOUT),
    ({"failed": True, "turn_exit_reason": "repeated_outer_errors"}, ERROR),
    ({"completed": False, "partial": True, "turn_exit_reason": "invalid_tool_calls"}, INCOMPLETE),
    (None, ERROR),
])
def test_classify(result, expected):
    assert classify_turn_exit(result) == expected


def test_budget_exit_code_is_distinct_for_oneshot():
    assert BUDGET_EXHAUSTED_EXIT_CODE not in (0, 1, 2, 75, 78, 130)
    assert turn_exit_code(_budget(), kanban_worker=False) == BUDGET_EXHAUSTED_EXIT_CODE
    assert is_budget_exhausted(_budget())


def test_kanban_worker_budget_keeps_failure_code():
    assert turn_exit_code(_budget(), kanban_worker=True) == 1


def test_failure_and_completion_codes_unchanged():
    assert turn_exit_code({"completed": True}, kanban_worker=False) == 0
    assert turn_exit_code({"failed": True, "completed": False}, kanban_worker=False) == 1
    assert turn_exit_code({**_budget(), "interrupted": True}, kanban_worker=False) == 130


def test_session_row_persists_last_turn_exit(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", source="cli")
        assert db.get_last_turn_exit("s1") is None
        db.set_last_turn_exit("s1", ITERATION_BUDGET, "max_iterations_reached(150/150)")
        got = db.get_last_turn_exit("s1")
        assert got["reason"] == ITERATION_BUDGET
        assert got["detail"] == "max_iterations_reached(150/150)"
        db.set_last_turn_exit("s1", COMPLETED, "text_response(stop)")
        assert db.get_last_turn_exit("s1")["reason"] == COMPLETED
    finally:
        db.close()


def test_finalizer_records_category_on_result_and_db():
    from types import SimpleNamespace
    from agent.turn_finalizer import _record_turn_exit
    calls = []
    db = SimpleNamespace(set_last_turn_exit=lambda *a: calls.append(a))
    agent = SimpleNamespace(_session_db=db, _persist_disabled=False, session_id="s1")
    result = _budget()
    _record_turn_exit(agent, result)
    assert result["turn_exit_category"] == ITERATION_BUDGET
    assert calls == [("s1", ITERATION_BUDGET, "max_iterations_reached(150/150)")]
    calls.clear()
    _record_turn_exit(SimpleNamespace(_session_db=db, _persist_disabled=True, session_id="s1"), _budget())
    assert calls == []
