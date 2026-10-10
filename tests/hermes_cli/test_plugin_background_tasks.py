from __future__ import annotations

import asyncio
import contextvars
import gc
import logging
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
