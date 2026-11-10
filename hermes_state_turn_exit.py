"""SessionDB mixin: the persisted reason the latest turn of a session ended.

Columns ``sessions.last_turn_exit_reason`` / ``_detail`` / ``_at`` (declared in
``hermes_state_common``; added to existing databases by column reconciliation).
"""

from __future__ import annotations

import time
from typing import Optional


class SessionTurnExitMixin:
    def set_last_turn_exit(self, session_id: str, reason: str, detail: Optional[str] = None) -> None:
        """Record why the latest turn of ``session_id`` ended (see ``agent.turn_exit_status``).

        ``reason`` is the coarse category (completed / iteration_budget / timeout / interrupted /
        error / incomplete); ``detail`` the raw loop ``turn_exit_reason``. Unlike ``end_reason``
        this is overwritten every turn, so a supervisor can tell a budget boundary from a crash."""
        self._write_sql(
            "UPDATE sessions SET last_turn_exit_reason = ?, last_turn_exit_detail = ?, "
            "last_turn_exit_at = ? WHERE id = ?",
            (reason, (detail or None) and str(detail)[:200], time.time(), session_id),
        )

    def get_last_turn_exit(self, session_id: str) -> Optional[dict]:
        """``{"reason", "detail", "at"}`` for the session's latest finished turn, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT last_turn_exit_reason, last_turn_exit_detail, last_turn_exit_at "
                "FROM sessions WHERE id = ?", (session_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return {"reason": row[0], "detail": row[1], "at": row[2]}
