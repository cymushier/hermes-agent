"""delegate_task result: an iteration-budget stop is flagged with the child's session id."""

from types import SimpleNamespace

from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry


def _child(**kw):
    return SimpleNamespace(model="m", session_id="child-sid", session_prompt_tokens=1,
                           session_completion_tokens=2, session_estimated_cost_usd=0.0, **kw)


def _entry(result):
    return _build_result_entry(_child(), result, 0, 1.0, _SchemaOutcome(None, None, [], 0))


def test_budget_stop_sets_flag_and_child_session_id():
    entry = _entry({"final_response": "partial summary", "completed": False,
                    "turn_exit_reason": "max_iterations_reached(250/250)"})
    assert entry["exit_reason"] == "max_iterations" and entry["truncated"] is True
    assert entry["budget_exhausted"] is True
    assert entry["child_session_id"] == "child-sid"


def test_completed_and_failed_children_are_not_budget_stops():
    done = _entry({"final_response": "ok", "completed": True})
    assert done["budget_exhausted"] is False and "child_session_id" not in done
    failed = _entry({"final_response": "boom", "failed": True, "completed": False})
    assert failed["budget_exhausted"] is False and "child_session_id" not in failed
