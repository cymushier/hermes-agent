from __future__ import annotations

import asyncio
import contextlib
import contextvars
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import run as gateway_run
from gateway.config import GatewayConfig
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.run_plugin_rewire import reload_plugins_verb
from gateway.run_startup import GatewayStartupMixin


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
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
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
    # Upstream's post-reload MCP reconcile re-discovers on its own thread/scope; out of scope here.
    monkeypatch.setattr("gateway.run_plugin_rewire._reconcile_plugin_mcp_servers", lambda runner, home: None)
    runner = SimpleNamespace(adapters={}, _served_profile_homes={})
    handler = reload_plugins_verb(runner, asyncio.get_running_loop())

    result = await asyncio.to_thread(handler, {"home": str(tmp_path)})

    assert result["reloaded"] is True
    assert observed == [(True, True)]
