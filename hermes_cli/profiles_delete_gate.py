"""Fail-closed multiplexer teardown gate for destructive profile deletion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_constants import clear_named_profile_deleted, mark_named_profile_deleted

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiplexerSnapshot:
    """Whether a live owner existed and the roster it reported before tombstoning."""

    owner_live: bool | None
    served_profiles: tuple[str, ...] | None


def notify_multiplexer(profile: str) -> Optional[list[str]]:
    from hermes_cli.gateway_multiplex_served import notify_multiplexer_profiles_changed
    return notify_multiplexer_profiles_changed(profile)


class ProfileTeardownPending(RuntimeError):
    """A live multiplexer may still own this profile, so destructive deletion did not start."""

    def __init__(self, profile: str, path: Path):
        self.profile = profile
        self.path = path
        self.retry_command = f"hermes profile delete {profile} --yes"
        super().__init__(
            f"Profile '{profile}' teardown is still pending because the live multiplexer did "
            f"not confirm it was unserved; the profile remains intact — retry with: "
            f"{self.retry_command}"
        )


def multiplexer_served_profiles() -> MultiplexerSnapshot:
    """Keep live-owner state distinct from an empty or unavailable roster."""
    try:
        from hermes_cli.gateway_multiplex_served import (
            live_default_gateway_pid,
            recorded_served_profiles,
        )
        if live_default_gateway_pid() is None:
            return MultiplexerSnapshot(False, ())
        roster = recorded_served_profiles()
        return MultiplexerSnapshot(True, None if roster is None else tuple(roster))
    except Exception:
        logger.debug("Could not determine live multiplexer profile ownership", exc_info=True)
        return MultiplexerSnapshot(None, None)


def require_confirmed_profile_unserve(
    profile: str,
    path: Path,
    snapshot: MultiplexerSnapshot,
    notify: Callable[[str], Optional[list[str]]],
    resnapshot: Callable[[], MultiplexerSnapshot],
) -> None:
    """Tombstone *profile* and refuse deletion unless any possible live owner confirms teardown."""
    mark_named_profile_deleted(path)
    served_after = notify(profile)
    post_snapshot = snapshot if served_after is not None else resnapshot()
    teardown_unconfirmed = (
        (served_after is None and post_snapshot.owner_live is not False)
        or (served_after is not None and profile in served_after)
    )
    if teardown_unconfirmed:
        clear_named_profile_deleted(path)
        raise ProfileTeardownPending(profile, path)
