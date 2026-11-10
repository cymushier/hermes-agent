"""Coarse, persisted classification of why a turn ended.

``turn_exit_reason`` is a free-form loop string (``text_response(stop)``,
``max_iterations_reached(150/150)`` ...). Supervisors, dashboards and the
budget-continuation paths need a small stable vocabulary instead, persisted on the
session row (``sessions.last_turn_exit_reason``) so it survives the process.
"""

from __future__ import annotations

from typing import Any

COMPLETED = "completed"
ITERATION_BUDGET = "iteration_budget"
TIMEOUT = "timeout"
INTERRUPTED = "interrupted"
ERROR = "error"
INCOMPLETE = "incomplete"

TURN_EXIT_CATEGORIES = frozenset({COMPLETED, ITERATION_BUDGET, TIMEOUT, INTERRUPTED, ERROR, INCOMPLETE})

_BUDGET_PREFIXES = ("max_iterations_reached(", "budget_exhausted")
_TIMEOUT_REASONS = frozenset({"timeout"})


def is_budget_exit_reason(reason: Any) -> bool:
    """True for a raw ``turn_exit_reason`` that means the iteration budget ran out."""
    return isinstance(reason, str) and reason.startswith(_BUDGET_PREFIXES)


def classify_turn_exit(result: Any) -> str:
    """Map a ``run_conversation`` result dict onto one of :data:`TURN_EXIT_CATEGORIES`."""
    if not isinstance(result, dict):
        return ERROR
    if result.get("interrupted"):
        return INTERRUPTED
    reason = result.get("turn_exit_reason")
    if result.get("failed"):
        if result.get("failure_reason") in _TIMEOUT_REASONS or "timeout" in str(reason or ""):
            return TIMEOUT
        return ERROR
    if is_budget_exit_reason(reason):
        return ITERATION_BUDGET
    if result.get("partial") or result.get("completed") is False:
        return INCOMPLETE
    return COMPLETED


def is_budget_exhausted(result: Any) -> bool:
    """True when a turn result is a resumable iteration-budget boundary."""
    return classify_turn_exit(result) == ITERATION_BUDGET
