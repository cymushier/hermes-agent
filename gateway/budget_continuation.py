"""Automatic continuation of gateway sessions whose turn stopped at the iteration budget.

Opt-in via ``agent.auto_continue_on_budget`` (off by default)::

    agent:
      auto_continue_on_budget:
        enabled: true
        max_continuations_per_session: 10
        message: null        # optional custom continuation prompt

After a gateway turn ends with the persisted exit category ``iteration_budget``
(``sessions.last_turn_exit_reason``), the runner queues one synthetic continuation turn
through the same path as ``/goal`` continuations, so a real user message still preempts it.
The per-session counter lives in ``state_meta`` (``budget_continue:<session_id>``) and resets on
every real user message. Interrupted, failed and approval-blocked turns never continue, and an
active ``/goal`` owns its own continuation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

MARKER_PREFIX = "[budget-continuation"
PRODUCER = "budget-continuation"
DEFAULT_MAX_CONTINUATIONS = 10
# The persisted exit row must belong to the turn that just ended.
_FRESHNESS_S = 600.0
DEFAULT_MESSAGE = (
    "The previous turn stopped because it reached the iteration budget, not because the work "
    "is done. Continue the task from where you left off (re-read your plan/checkpoint if "
    "needed). If everything is already complete, reply with a short final summary instead."
)


@dataclass(frozen=True)
class BudgetContinuationConfig:
    enabled: bool = False
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS
    message: str = DEFAULT_MESSAGE


def parse_config(cfg: Any) -> BudgetContinuationConfig:
    """``agent.auto_continue_on_budget`` → config; ``true``/``false`` shorthand accepted."""
    if isinstance(cfg, bool):
        return BudgetContinuationConfig(enabled=cfg)
    if not isinstance(cfg, dict):
        return BudgetContinuationConfig()
    try:
        cap = int(cfg.get("max_continuations_per_session", DEFAULT_MAX_CONTINUATIONS))
    except (TypeError, ValueError):
        cap = DEFAULT_MAX_CONTINUATIONS
    msg = cfg.get("message")
    return BudgetContinuationConfig(
        enabled=bool(cfg.get("enabled", False)),
        max_continuations=max(0, cap),
        message=msg.strip() if isinstance(msg, str) and msg.strip() else DEFAULT_MESSAGE,
    )


def load_config() -> BudgetContinuationConfig:
    try:
        from hermes_cli.config import load_config as _load
        return parse_config(((_load() or {}).get("agent") or {}).get("auto_continue_on_budget"))
    except Exception:  # health: allow BLE001 -- unreadable config must mean "off", never a broken turn; logged
        logger.debug("budget continuation: config unreadable; treating as disabled", exc_info=True)
        return BudgetContinuationConfig()


def is_continuation_text(text: Any) -> bool:
    return isinstance(text, str) and text.lstrip().startswith(MARKER_PREFIX)


def is_continuation_event(event: Any) -> bool:
    meta = getattr(event, "metadata", None) or {}
    return bool(meta.get(PRODUCER)) or is_continuation_text(getattr(event, "text", None))


def build_prompt(cfg: BudgetContinuationConfig, n: int) -> str:
    """The visible transcript marker plus the continuation instruction."""
    return f"{MARKER_PREFIX} {n}/{cfg.max_continuations}] {cfg.message}"


def _counter_key(session_id: str) -> str:
    return f"budget_continue:{session_id}"


def read_count(db: Any, session_id: str) -> int:
    try:
        return int(db.get_meta(_counter_key(session_id)) or 0)
    except (TypeError, ValueError):
        return 0


def write_count(db: Any, session_id: str, n: int) -> None:
    db.set_meta(_counter_key(session_id), str(int(n)))


def is_fresh_budget_exit(last_exit: Optional[dict], now: Optional[float] = None) -> bool:
    if not isinstance(last_exit, dict) or last_exit.get("reason") != "iteration_budget":
        return False
    at = last_exit.get("at")
    now = time.time() if now is None else now
    return isinstance(at, (int, float)) and (now - at) <= _FRESHNESS_S


def decide(
    cfg: BudgetContinuationConfig, *, db: Any, session_id: str, real_user_turn: bool,
    continuation_turn: bool, failed: bool, goal_active: bool, approval_pending: bool,
    now: Optional[float] = None,
) -> Optional[str]:
    """Return the continuation prompt to enqueue, or None. Maintains the per-session counter."""
    if not session_id or db is None:
        return None
    if real_user_turn:
        # A human spoke: the budget streak starts over.
        if read_count(db, session_id):
            write_count(db, session_id, 0)
    if not cfg.enabled or cfg.max_continuations <= 0:
        return None
    if not (real_user_turn or continuation_turn):
        return None  # heartbeats, /loop ticks, goal continuations own their own follow-up
    if failed or goal_active or approval_pending:
        return None
    last = db.get_last_turn_exit(session_id) if hasattr(db, "get_last_turn_exit") else None
    if not is_fresh_budget_exit(last, now):  # completed, interrupted, error, timeout, ...
        return None
    n = read_count(db, session_id) + 1
    if n > cfg.max_continuations:
        return None
    write_count(db, session_id, n)
    return build_prompt(cfg, n)
