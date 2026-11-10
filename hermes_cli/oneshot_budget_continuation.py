"""``chat -q --continue-on-budget N``: keep a one-shot run going past the iteration budget.

When a one-shot turn stops at ``agent.max_turns`` (exit category ``iteration_budget``), the run
sends up to N continuation turns IN THE SAME SESSION, so a supervised worker keeps its full
context instead of being respawned from a resume brief. Each continuation is a visible
``[budget-continuation n/N]`` user turn. Interrupted or failed turns stop immediately; the
process exit code is the final turn's (3 when the last continuation still hit the budget).

Also settable with ``HERMES_CONTINUE_ON_BUDGET=N`` (consumed, so tool subprocesses and nested
``hermes`` runs do not inherit it). Kanban workers are excluded: the dispatcher owns their
budget semantics (``goal_mode`` / ``timed_out``).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

ENV_VAR = "HERMES_CONTINUE_ON_BUDGET"
MAX_CONTINUATIONS = 1000
MARKER_PREFIX = "[budget-continuation"
PROMPT = (
    "The previous turn stopped because it reached the iteration budget, not because the work "
    "is done. Continue the task from where you left off (re-read your plan/checkpoint if "
    "needed). If everything is already complete, reply with a short final summary instead."
)


def take_continue_on_budget(explicit: Optional[int] = None) -> int:
    """Resolve the continuation cap (flag beats env) and consume the env var."""
    raw = os.environ.pop(ENV_VAR, None)
    value: Any = explicit if explicit is not None else raw
    try:
        n = int(str(value).strip()) if value is not None and str(value).strip() else 0
    except ValueError:
        return 0
    return max(0, min(n, MAX_CONTINUATIONS))


def continuation_prompt(n: int, cap: int) -> str:
    return f"{MARKER_PREFIX} {n}/{cap}] {PROMPT}"


def run_budget_continuations(
    run_turn: Callable[[str], Any], last_result: Callable[[], Any], cap: int,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """Re-enter the same session while the latest turn ended on the budget. Returns turns sent."""
    from agent.turn_exit_status import is_budget_exhausted

    sent = 0
    while sent < cap and is_budget_exhausted(last_result()):
        sent += 1
        prompt = continuation_prompt(sent, cap)
        if log is not None:
            log(f"{MARKER_PREFIX} {sent}/{cap}] iteration budget reached; continuing in the same session")
        run_turn(prompt)
    return sent
