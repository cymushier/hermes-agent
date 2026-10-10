from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.profiles_delete_gate import MultiplexerSnapshot


@pytest.fixture
def deletable_profile(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / ".env").write_text("# profile identity\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda *_: False)
    monkeypatch.setattr(profiles, "_cleanup_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_maybe_unregister_gateway_service", lambda *_: None)
    monkeypatch.setattr(profiles, "_stop_profile_backends", lambda *_: None)
    monkeypatch.setattr(profiles, "_stop_bot_desktop", lambda *_: None)
    monkeypatch.setattr(profiles, "_get_wrapper_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda **_: None)
    return profile


def test_delete_keeps_served_profile_when_multiplexer_cannot_confirm_unserve(
    deletable_profile, monkeypatch
):
    from hermes_cli.profiles import ProfileTeardownPending
    from hermes_constants import named_profile_is_deleted

    monkeypatch.setattr(
        profiles, "_multiplexer_served_profiles",
        lambda: MultiplexerSnapshot(True, ("default", "worker"))
    )
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    monkeypatch.setattr(
        profiles, "_purge_identity", lambda *_: pytest.fail("identity purged before teardown")
    )

    with pytest.raises(ProfileTeardownPending, match="teardown is still pending") as exc:
        profiles.delete_profile("worker", yes=True)

    assert exc.value.retry_command == "hermes profile delete worker --yes"
    assert deletable_profile.is_dir()
    assert (deletable_profile / ".env").is_file()
    assert not named_profile_is_deleted(deletable_profile)


def test_delete_keeps_profile_when_live_multiplexer_roster_is_unavailable(
    deletable_profile, monkeypatch
):
    from hermes_cli.profiles import ProfileTeardownPending

    monkeypatch.setattr(
        profiles, "_multiplexer_served_profiles", lambda: MultiplexerSnapshot(None, None)
    )
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    monkeypatch.setattr(
        profiles, "_purge_identity", lambda *_: pytest.fail("identity purged before teardown")
    )

    with pytest.raises(ProfileTeardownPending):
        profiles.delete_profile("worker", yes=True)

    assert deletable_profile.is_dir()


def test_delete_allows_unavailable_multiplexer_when_profile_was_not_served(
    deletable_profile, monkeypatch
):
    monkeypatch.setattr(
        profiles, "_multiplexer_served_profiles", lambda: MultiplexerSnapshot(False, ())
    )
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    monkeypatch.setattr(profiles, "_purge_identity", lambda *_: True)

    removed = profiles.delete_profile("worker", yes=True)

    assert removed == deletable_profile
    assert not deletable_profile.exists()


def test_delete_refuses_owner_that_appears_after_offline_snapshot(
    deletable_profile, monkeypatch
):
    from hermes_cli.profiles import ProfileTeardownPending
    from hermes_constants import named_profile_is_deleted

    monkeypatch.setattr(
        profiles, "_multiplexer_served_profiles", lambda: MultiplexerSnapshot(False, ())
    )
    monkeypatch.setattr(
        profiles, "_notify_multiplexer", lambda *_: ["default", "worker"]
    )
    monkeypatch.setattr(
        profiles, "_purge_identity", lambda *_: pytest.fail("identity purged beneath live owner")
    )

    with pytest.raises(ProfileTeardownPending):
        profiles.delete_profile("worker", yes=True)

    assert deletable_profile.is_dir()
    assert not named_profile_is_deleted(deletable_profile)


def test_delete_refuses_ambiguous_notification_when_owner_appears_after_snapshot(
    deletable_profile, monkeypatch
):
    from hermes_cli.profiles import ProfileTeardownPending
    from hermes_constants import named_profile_is_deleted

    snapshots = iter(
        [MultiplexerSnapshot(False, ()), MultiplexerSnapshot(True, None)]
    )
    monkeypatch.setattr(
        profiles, "_multiplexer_served_profiles", lambda: next(snapshots)
    )
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    monkeypatch.setattr(
        profiles, "_purge_identity", lambda *_: pytest.fail("identity purged beneath new owner")
    )

    with pytest.raises(ProfileTeardownPending):
        profiles.delete_profile("worker", yes=True)

    assert deletable_profile.is_dir()
    assert not named_profile_is_deleted(deletable_profile)


def test_delete_requires_post_tombstone_confirmation_from_live_owner_even_if_initially_unserved(
    deletable_profile, monkeypatch
):
    from hermes_cli.profiles import ProfileTeardownPending

    monkeypatch.setattr(
        profiles, "_multiplexer_served_profiles",
        lambda: MultiplexerSnapshot(True, ("default", "other"))
    )
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda *_: None)
    monkeypatch.setattr(
        profiles, "_purge_identity", lambda *_: pytest.fail("identity purged before teardown")
    )

    with pytest.raises(ProfileTeardownPending):
        profiles.delete_profile("worker", yes=True)

    assert deletable_profile.is_dir()


@pytest.mark.asyncio
async def test_cancellation_resistant_observer_cannot_be_deleted_before_unserve_confirmation(
    deletable_profile, monkeypatch
):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_cli.profiles import ProfileTeardownPending
    from hermes_constants import named_profile_is_deleted

    manager = PluginManager(scope_key=str(deletable_profile))
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def observer():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

    PluginContext(
        PluginManifest(name="approval-observer", source="user"), manager
    ).register_background_task("observer", observer)
    loop = asyncio.get_running_loop()
    manager.bind_runtime_loop(loop)
    await asyncio.wait_for(started.wait(), timeout=1)

    def notify(_name):
        future = asyncio.run_coroutine_threadsafe(
            manager.unbind_runtime_loop(loop, timeout=0), loop
        )
        try:
            future.result(timeout=1)
        except TimeoutError:
            return None
        return ["default"]

    monkeypatch.setattr(
        profiles, "_multiplexer_served_profiles",
        lambda: MultiplexerSnapshot(True, ("default", "worker"))
    )
    monkeypatch.setattr(profiles, "_notify_multiplexer", notify)
    monkeypatch.setattr(
        profiles, "_purge_identity", lambda *_: pytest.fail("identity purged before teardown")
    )

    with pytest.raises(ProfileTeardownPending):
        await asyncio.to_thread(profiles.delete_profile, "worker", yes=True)

    assert cancellation_seen.is_set()
    assert deletable_profile.is_dir()
    assert not named_profile_is_deleted(deletable_profile)

    release.set()
    await manager.unbind_runtime_loop(loop, timeout=1)