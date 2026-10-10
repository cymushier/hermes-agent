from __future__ import annotations

import asyncio
import contextlib
import contextvars
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.run_plugin_rewire import reload_plugins_verb
from gateway.run_profile_reconcile import GatewayProfileReconcileMixin
from gateway.run_startup import GatewayStartupMixin
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.mark.asyncio
async def test_primary_plugin_runtime_loop_is_bound_inside_launch_profile_scope(monkeypatch, tmp_path):
    in_scope = contextvars.ContextVar("primary_plugin_scope", default=False)
    manager = SimpleNamespace(bind_runtime_loop=lambda loop: observed.append((loop, in_scope.get())))
    observed = []
    scope_calls = []

    @contextlib.contextmanager
    def scope(home, prepared_secret_scope=None, *, hydrate_secrets=True):
        scope_calls.append((Path(home), prepared_secret_scope, hydrate_secrets))
        token = in_scope.set(True)
        try:
            yield
        finally:
            in_scope.reset(token)

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", scope)
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    GatewayStartupMixin._bind_launch_profile_plugin_runtime()

    assert observed == [(asyncio.get_running_loop(), True)]
    assert scope_calls == [(tmp_path, None, False)]


@pytest.mark.asyncio
async def test_secondary_discovery_binds_runtime_loop_inside_profile_scope(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "secondary"
    profile_home.mkdir(parents=True)
    in_scope = contextvars.ContextVar("secondary_plugin_scope", default=False)
    observed = []
    manager = SimpleNamespace(
        scope_key=str(profile_home),
        bind_runtime_loop=lambda loop: observed.append((loop, in_scope.get())),
    )

    @contextlib.contextmanager
    def scope(home, prepared_secret_scope=None, *, hydrate_secrets=True):
        assert Path(home) == profile_home
        assert hydrate_secrets is False
        token = in_scope.set(True)
        try:
            yield
        finally:
            in_scope.reset(token)

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", scope)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", dict)
    monkeypatch.setattr(gateway_run, "_own_policy_open_startup_violation", lambda config: None)
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda home: {})
    monkeypatch.setattr("gateway.run_startup.recover_left_core_in", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig())

    runner = SimpleNamespace(
        _subscribe_plugin_rewire=lambda *args: None,
        _register_config_hooks=lambda *args, **kwargs: None,
        _snapshot_profile_busy_modes=lambda *args: None,
    )
    result = await GatewayAdapterLifecycleMixin._load_secondary_profile_config(
        runner, "secondary", profile_home
    )

    assert isinstance(result, GatewayConfig)
    assert observed == [(asyncio.get_running_loop(), True)]


@pytest.mark.asyncio
async def test_secondary_policy_rejection_unbinds_runtime_tasks(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "secondary"
    profile_home.mkdir(parents=True)
    manager = PluginManager(scope_key=str(profile_home))
    factory_calls = []

    async def observer():
        factory_calls.append("broker-created")
        await asyncio.Event().wait()

    PluginContext(PluginManifest(name="approval-observer", source="user"), manager).register_background_task(
        "observer", observer
    )

    monkeypatch.setattr(gateway_run, "_load_gateway_config", dict)
    monkeypatch.setattr(gateway_run, "_own_policy_open_startup_violation", lambda config: "open DM policy")
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda home: {})
    monkeypatch.setattr("gateway.run_startup.recover_left_core_in", lambda *args, **kwargs: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: GatewayConfig())
    runner = SimpleNamespace(
        _subscribe_plugin_rewire=lambda *args: None,
        _register_config_hooks=lambda *args, **kwargs: None,
        _snapshot_profile_busy_modes=lambda *args: None,
    )

    with pytest.raises(gateway_run.MultiplexConfigError):
        await GatewayAdapterLifecycleMixin._load_secondary_profile_config(
            runner, "secondary", profile_home
        )
    await asyncio.sleep(0)

    assert manager._runtime_loop is None
    assert factory_calls == []
    assert len(manager._background_task_factories) == 1


@pytest.mark.asyncio
async def test_unserve_cancels_profile_task_before_adapter_teardown_and_reserve_restarts_once(
    monkeypatch, tmp_path
):
    profile_home = tmp_path / "profiles" / "secondary"
    profile_home.mkdir(parents=True)
    runtime_scope = contextvars.ContextVar("secondary_runtime_generation", default="outside")
    manager = PluginManager(scope_key=str(profile_home))
    starts = []
    cancelled = asyncio.Event()
    started = asyncio.Event()
    order = []

    async def observer():
        starts.append(runtime_scope.get())
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            order.append("task-cancelled")
            cancelled.set()
            raise

    PluginContext(PluginManifest(name="approval-observer", source="user"), manager).register_background_task(
        "observer", observer
    )
    loop = asyncio.get_running_loop()
    token = runtime_scope.set("old-profile-secret")
    try:
        manager.bind_runtime_loop(loop)
    finally:
        runtime_scope.reset(token)
    await asyncio.wait_for(started.wait(), timeout=1)

    @contextlib.contextmanager
    def scope(home, prepared_secret_scope=None, *, hydrate_secrets=True):
        assert Path(home) == profile_home
        yield

    async def adapter_teardown(adapter, platform, profile=None):
        assert cancelled.is_set()
        order.append("adapter-teardown")

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", scope)
    monkeypatch.setattr(gateway_run, "_write_runtime_status_quiet", lambda **kwargs: None)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_state_registry.close_all_under", lambda home: None)
    memory_store = SimpleNamespace(release_all_under=lambda home: None)
    monkeypatch.setattr(
        "plugins.memory.import_provider_module",
        lambda provider, module: SimpleNamespace(MemoryStore=memory_store),
    )
    monkeypatch.setattr("tools.mcp_tool_lifecycle.shutdown_mcp_servers", lambda **kwargs: None)
    monkeypatch.setattr("hermes_logging.release_profile_log_handlers", lambda home: None)
    runner = SimpleNamespace(
        config=SimpleNamespace(_runtime_profile_homes=(("secondary", profile_home),)),
        _profile_failed_platforms={},
        _profile_adapters={"secondary": {Platform.TELEGRAM: object()}},
        pairing_stores={},
        _busy_text_modes_by_profile={},
        _busy_input_modes_by_profile={},
        _busy_text_timing_by_profile={},
        _human_delay_by_profile={},
        _profile_configs={},
        _served_profile_homes={"secondary": profile_home},
        _served_profile_signatures={"secondary": ()},
        _agent_cache={},
        _adapter_disconnect_timeout_secs=lambda: 1.0,
        _bounded_adapter_teardown=adapter_teardown,
        _adapter_claims=lambda platform, adapter: (),
        _hold_secondary_claims=lambda name, platform, adapter, claims: None,
        _release_secondary_claims=lambda name, platform, adapter: None,
        _secondary_held_claims={},
        _evict_cached_agent=lambda key: None,
    )

    await GatewayProfileReconcileMixin._unserve_profile(runner, "secondary", profile_home)
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    assert order == ["task-cancelled", "adapter-teardown"]
    assert manager._runtime_loop is None
    await asyncio.sleep(0)
    assert starts == ["old-profile-secret"]  # no broker/task recreation after teardown

    # Re-serving binds the retained factory under the new profile scope exactly once.
    started.clear()
    token = runtime_scope.set("new-profile-secret")
    try:
        manager.bind_runtime_loop(loop)
        manager.bind_runtime_loop(loop)
    finally:
        runtime_scope.reset(token)
    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert starts == ["old-profile-secret", "new-profile-secret"]
    await manager.unbind_runtime_loop(loop)


@pytest.mark.asyncio
async def test_primary_force_reload_runs_inside_registration_scope(monkeypatch, tmp_path):
    in_scope = contextvars.ContextVar("primary_reload_scope", default=False)
    manager = SimpleNamespace(_plugins={"probe": object()})
    observed = []

    @contextlib.contextmanager
    def scope(home, prepared_secret_scope=None, *, hydrate_secrets=True):
        assert Path(home) == tmp_path
        assert hydrate_secrets is False
        token = in_scope.set(True)
        try:
            yield
        finally:
            in_scope.reset(token)

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", scope)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.plugins.discover_plugins",
        lambda force=False: observed.append((force, in_scope.get())),
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.plugins_activation.activation_summaries", lambda value: [])
    monkeypatch.setattr("gateway.run_plugin_rewire._reconcile_plugin_mcp_servers", lambda *args: None)
    runner = SimpleNamespace(adapters={}, _served_profile_homes={})
    handler = reload_plugins_verb(runner, asyncio.get_running_loop())

    result = await asyncio.to_thread(handler, {"home": str(tmp_path)})

    assert result["reloaded"] is True
    assert observed == [(True, True)]
