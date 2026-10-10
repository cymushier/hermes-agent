from __future__ import annotations

import asyncio
import contextvars
import gc
import logging
import threading
import warnings

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context(manager: PluginManager, name: str = "runtime-probe") -> PluginContext:
    return PluginContext(PluginManifest(name=name, source="user"), manager)


def test_sync_registration_is_inert_until_runtime_loop_is_bound():
    manager = PluginManager()
    ctx = _context(manager)
    calls = []

    async def worker():
        calls.append("started")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handle = ctx.register_background_task("observer", worker)
        gc.collect()

    assert handle.active
    assert calls == []
    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]
    assert manager.unload("runtime-probe") is True


@pytest.mark.asyncio
async def test_pending_registration_starts_once_with_bind_context():
    runtime_scope = contextvars.ContextVar("runtime_scope", default="outside")
    manager = PluginManager()
    ctx = _context(manager)
    observed = []
    started = asyncio.Event()

    async def worker():
        observed.append(runtime_scope.get())
        started.set()

    token = runtime_scope.set("discovery")
    try:
        ctx.register_background_task("observer", worker)
    finally:
        runtime_scope.reset(token)

    token = runtime_scope.set("bound-runtime")
    try:
        loop = asyncio.get_running_loop()
        manager.bind_runtime_loop(loop)
        manager.bind_runtime_loop(loop)
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(0)
    finally:
        runtime_scope.reset(token)

    assert observed == ["bound-runtime"]


@pytest.mark.asyncio
async def test_late_registration_starts_on_bound_loop_with_registration_context():
    runtime_scope = contextvars.ContextVar("late_runtime_scope", default="outside")
    manager = PluginManager()
    manager.bind_runtime_loop(asyncio.get_running_loop())
    ctx = _context(manager)
    observed = []
    started = asyncio.Event()

    async def worker():
        observed.append((runtime_scope.get(), asyncio.get_running_loop()))
        started.set()

    token = runtime_scope.set("late-profile")
    try:
        await asyncio.to_thread(ctx.register_background_task, "observer", worker)
    finally:
        runtime_scope.reset(token)

    await asyncio.wait_for(started.wait(), timeout=1)
    assert observed == [("late-profile", asyncio.get_running_loop())]


@pytest.mark.asyncio
async def test_unload_cancels_task_and_replacement_starts_once():
    manager = PluginManager()
    manager.bind_runtime_loop(asyncio.get_running_loop())
    manifest = PluginManifest(name="replaceable", source="user")
    old_started = asyncio.Event()
    old_cancelled = asyncio.Event()
    replacement_started = asyncio.Event()
    starts = []

    async def old_worker():
        starts.append("old")
        old_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            old_cancelled.set()
            raise

    async def replacement_worker():
        starts.append("replacement")
        replacement_started.set()

    PluginContext(manifest, manager).register_background_task("observer", old_worker)
    await asyncio.wait_for(old_started.wait(), timeout=1)
    assert manager.unload("replaceable") is True
    await asyncio.wait_for(old_cancelled.wait(), timeout=1)

    PluginContext(manifest, manager).register_background_task("observer", replacement_worker)
    await asyncio.wait_for(replacement_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert starts == ["old", "replacement"]


@pytest.mark.asyncio
async def test_force_reload_cancels_generation_and_starts_replacement(monkeypatch):
    from hermes_cli.plugins import LoadedPlugin

    manager = PluginManager()
    manifest = PluginManifest(name="force-reload", source="user")
    starts = []
    cancelled = asyncio.Event()
    replacement_started = asyncio.Event()
    generation = 0

    def discover_generation():
        nonlocal generation
        generation += 1
        current = generation

        async def worker():
            starts.append(current)
            if current == 2:
                replacement_started.set()
                return
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        PluginContext(manifest, manager).register_background_task("observer", worker)
        manager._plugins["force-reload"] = LoadedPlugin(manifest=manifest, enabled=True)

    monkeypatch.setattr(manager, "_discover_and_load_inner", discover_generation)
    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", lambda: None)
    monkeypatch.setattr(manager, "_re_register_config_hooks_after_force", lambda: None)

    manager.discover_and_load()
    manager.bind_runtime_loop(asyncio.get_running_loop())
    await asyncio.sleep(0)
    manager.discover_and_load(force=True)

    await asyncio.wait_for(cancelled.wait(), timeout=1)
    await asyncio.wait_for(replacement_started.wait(), timeout=1)
    assert starts == [1, 2]


def test_bind_runtime_loop_rejects_stopped_loop():
    manager = PluginManager()
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="running host asyncio loop"):
            manager.bind_runtime_loop(loop)
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_factory_failure_is_reported(caplog):
    manager = PluginManager()
    manager.bind_runtime_loop(asyncio.get_running_loop())

    def invalid_factory():
        return None

    with caplog.at_level(logging.ERROR, logger="hermes_cli.plugins"):
        _context(manager, "invalid-factory").register_background_task("observer", invalid_factory)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert "background task observer failed" in caplog.text
    assert "factory must return a coroutine" in caplog.text


@pytest.mark.asyncio
async def test_profile_managers_bind_tasks_to_distinct_secret_scopes(tmp_path):
    from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    homes = [tmp_path / "alpha", tmp_path / "beta"]
    managers = [PluginManager(scope_key=str(home)) for home in homes]
    observed = []
    done = asyncio.Event()

    for index, manager in enumerate(managers):
        async def worker(index=index):
            observed.append((index, get_hermes_home(), get_secret("RUNTIME_TASK_SECRET")))
            if len(observed) == 2:
                done.set()

        _context(manager, f"profile-{index}").register_background_task("observer", worker)

    loop = asyncio.get_running_loop()
    for home, manager, secret in zip(homes, managers, ("alpha-secret", "beta-secret")):
        home.mkdir()
        home_token = set_hermes_home_override(home)
        secret_token = set_secret_scope(
            {"RUNTIME_TASK_SECRET": secret}, profile_home=str(home)
        )
        try:
            manager.bind_runtime_loop(loop)
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    await asyncio.wait_for(done.wait(), timeout=1)
    assert sorted(observed) == [
        (0, homes[0], "alpha-secret"),
        (1, homes[1], "beta-secret"),
    ]


@pytest.mark.asyncio
async def test_unbind_cancels_owner_tasks_and_rebind_starts_one_fresh_generation():
    runtime_scope = contextvars.ContextVar("rebind_runtime_scope", default="outside")
    manager = PluginManager()
    starts = []
    cancelled = []
    started = asyncio.Event()

    async def worker():
        generation = len(starts) + 1
        starts.append((generation, runtime_scope.get()))
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(generation)
            raise

    _context(manager).register_background_task("observer", worker)
    loop = asyncio.get_running_loop()

    token = runtime_scope.set("first-profile-scope")
    try:
        manager.bind_runtime_loop(loop)
    finally:
        runtime_scope.reset(token)
    await asyncio.wait_for(started.wait(), timeout=1)

    await manager.unbind_runtime_loop(loop)
    await manager.unbind_runtime_loop(loop)  # idempotent after teardown
    assert cancelled == [1]
    assert manager._runtime_loop is None
    assert len(manager._background_task_factories) == 1

    started.clear()
    token = runtime_scope.set("fresh-profile-scope")
    try:
        manager.bind_runtime_loop(loop)
        manager.bind_runtime_loop(loop)
    finally:
        runtime_scope.reset(token)
    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert starts == [(1, "first-profile-scope"), (2, "fresh-profile-scope")]
    await manager.unbind_runtime_loop(loop)
    assert cancelled == [1, 2]


@pytest.mark.asyncio
async def test_unbind_rejects_wrong_loop_without_disturbing_bound_tasks():
    manager = PluginManager()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def worker():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    _context(manager).register_background_task("observer", worker)
    bound_loop = asyncio.get_running_loop()
    manager.bind_runtime_loop(bound_loop)
    await asyncio.wait_for(started.wait(), timeout=1)
    wrong_loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="different runtime loop"):
            await manager.unbind_runtime_loop(wrong_loop)
        assert manager._runtime_loop is bound_loop
        assert not cancelled.is_set()
    finally:
        wrong_loop.close()
        await manager.unbind_runtime_loop(bound_loop)


@pytest.mark.asyncio
async def test_unbinding_one_profile_manager_does_not_cancel_primary():
    primary = PluginManager(scope_key="primary")
    secondary = PluginManager(scope_key="secondary")
    primary_started = asyncio.Event()
    secondary_started = asyncio.Event()
    primary_cancelled = asyncio.Event()
    secondary_cancelled = asyncio.Event()

    async def primary_worker():
        primary_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            primary_cancelled.set()
            raise

    async def secondary_worker():
        secondary_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            secondary_cancelled.set()
            raise

    _context(primary, "primary-plugin").register_background_task("observer", primary_worker)
    _context(secondary, "secondary-plugin").register_background_task("observer", secondary_worker)
    loop = asyncio.get_running_loop()
    primary.bind_runtime_loop(loop)
    secondary.bind_runtime_loop(loop)
    await asyncio.wait_for(asyncio.gather(primary_started.wait(), secondary_started.wait()), timeout=1)

    await secondary.unbind_runtime_loop(loop)
    assert secondary_cancelled.is_set()
    assert not primary_cancelled.is_set()
    assert primary._runtime_loop is loop

    await primary.unbind_runtime_loop(loop)


@pytest.mark.asyncio
async def test_unbind_from_another_thread_waits_on_owner_loop():
    manager = PluginManager()
    started = threading.Event()
    cancelled = threading.Event()

    async def worker():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    _context(manager).register_background_task("observer", worker)
    owner_loop = asyncio.new_event_loop()
    owner_thread = threading.Thread(target=owner_loop.run_forever, daemon=True)
    owner_thread.start()
    try:
        async def bind_on_owner():
            manager.bind_runtime_loop(owner_loop)
            await asyncio.sleep(0)

        asyncio.run_coroutine_threadsafe(bind_on_owner(), owner_loop).result(timeout=1)
        assert await asyncio.to_thread(started.wait, 1)

        await manager.unbind_runtime_loop(owner_loop)

        assert cancelled.is_set()
        assert manager._runtime_loop is None
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        await asyncio.to_thread(owner_thread.join, 1)
        owner_loop.close()


@pytest.mark.asyncio
async def test_cancellation_timeout_blocks_rebind_and_repeated_unbind_joins_stopping_task():
    manager = PluginManager()
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_worker():
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

    _context(manager).register_background_task("observer", stubborn_worker)
    loop = asyncio.get_running_loop()
    manager.bind_runtime_loop(loop)
    await asyncio.wait_for(started.wait(), timeout=1)

    with pytest.raises(TimeoutError, match="profile teardown cannot finish"):
        await manager.unbind_runtime_loop(loop, timeout=0)
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
    with pytest.raises(RuntimeError, match="still stopping"):
        manager.bind_runtime_loop(loop)

    release.set()
    await manager.unbind_runtime_loop(loop, timeout=1)
    manager.bind_runtime_loop(loop)
    await manager.unbind_runtime_loop(loop)
