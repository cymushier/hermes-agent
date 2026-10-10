"""Host-loop lifecycle for plugin-owned runtime background-task factories.

Mixed into :class:`hermes_cli.plugins.PluginManager`; the public registration entry point remains
``PluginContext.register_background_task``.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from hermes_cli.plugins_ledger import PluginRegistration
from hermes_cli.plugins_manifest import PluginManifest, manifest_key

if TYPE_CHECKING:  # pragma: no cover
    pass

logger = logging.getLogger("hermes_cli.plugins")

_RUNTIME_TASK_CANCEL_TIMEOUT_SECS = 5.0


@dataclass
class _RegisteredBackgroundTask:
    """A plugin-owned coroutine factory, pending or running on the host loop."""

    plugin_key: str
    name: str
    factory: Callable[[], Any]
    registration_context: contextvars.Context
    handle: Optional[PluginRegistration] = None
    task: Optional[asyncio.Task] = None
    scheduled: bool = False
    active: bool = True


class PluginContextRuntimeTasksMixin:
    """Public cleanup and async-work methods mixed into ``PluginContext``."""

    def on_unload(self, callback: Callable[[], None]) -> PluginRegistration:
        """Register cleanup, run in reverse acquisition order among registration teardown."""
        if not callable(callback):
            raise TypeError("on_unload callback must be callable")
        handle = self._track("on_unload", getattr(callback, "__name__", "callback"), callback)
        logger.debug("Plugin %s registered on_unload callback", self.manifest.name)
        return handle

    def spawn_task(self, coro: Any, *, name: Optional[str] = None) -> Any:
        """Spawn supervised work when plugin code already has a running host loop."""
        if not asyncio.iscoroutine(coro):
            raise TypeError("spawn_task expects a coroutine")
        with self._task_lifecycle_lock:
            if self._load_abandoned:
                coro.close()
                raise RuntimeError(
                    "cannot spawn a background task after the plugin load was abandoned"
                )
            loop = self._task_loop
            if loop is None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    coro.close()
                    raise RuntimeError(
                        "ctx.spawn_task requires a running host asyncio loop; "
                        "synchronous plugin discovery cannot supervise background tasks"
                    ) from None
            if loop.is_closed() or not loop.is_running():
                coro.close()
                raise RuntimeError("ctx.spawn_task cannot use a stopped host asyncio loop")
            task_name = name or f"plugin:{self.plugin_id}:task"
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            try:
                if loop is current_loop:
                    task = loop.create_task(coro, name=task_name)
                else:
                    task = asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError as exc:
                coro.close()
                raise RuntimeError(
                    f"ctx.spawn_task could not schedule on the host asyncio loop: {exc}"
                ) from None

            def _cancel_task() -> None:
                if not task.done():
                    task.cancel()

            try:
                handle = self._track("background_task", task_name, _cancel_task)
            except BaseException:
                task.cancel()
                raise

        def _task_done(completed: Any) -> None:
            try:
                if not completed.cancelled() and (exc := completed.exception()) is not None:
                    logger.error(
                        "Plugin %s background task %s failed: %s",
                        self.manifest.name, task_name, exc,
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
            finally:
                handle.dispose()

        task.add_done_callback(_task_done)
        logger.debug("Plugin %s spawned supervised task: %s", self.manifest.name, task_name)
        return task

    def register_background_task(
        self, name: str, factory: Callable[[], Any],
    ) -> PluginRegistration:
        """Retain zero-argument async work without creating a coroutine during discovery."""
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("background task name must not be empty")
        if not callable(factory):
            raise TypeError("background task factory must be callable")
        with self._task_lifecycle_lock:
            if self._load_abandoned:
                raise RuntimeError(
                    "cannot register a background task after the plugin load was abandoned"
                )
            return self._manager._register_background_task(
                self.manifest, clean_name, factory, contextvars.copy_context()
            )


class PluginRuntimeTasksMixin:
    def _init_runtime_tasks(self) -> None:
        # The loop survives force rediscovery so replacement registrations start immediately; each
        # factory/Task remains ledger-owned by its plugin generation.
        self._runtime_loop: Optional[asyncio.AbstractEventLoop] = None
        self._runtime_task_lock = threading.RLock()
        self._background_task_factories: Dict[Tuple[str, str], _RegisteredBackgroundTask] = {}
        # Detached generations stay here until their cancellation settles. A new bind is refused
        # while any remain: profile teardown must not overlap a stale secret-bearing Context with a
        # fresh generation for the same manager.
        self._runtime_tasks_stopping: set[asyncio.Task] = set()

    def bind_runtime_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind this profile's manager to its running host loop and start pending factories once.

        Call on the gateway loop while the owning profile runtime/secret scope is active. Rebinding the
        same loop is idempotent; binding a stopped or different loop raises visibly.
        """
        if loop.is_closed() or not loop.is_running():
            raise RuntimeError("bind_runtime_loop requires a running host asyncio loop")
        bind_context = contextvars.copy_context()
        with self._runtime_task_lock:
            if self._runtime_loop is not None and self._runtime_loop is not loop:
                raise RuntimeError("plugin manager is already bound to a different runtime loop")
            self._runtime_tasks_stopping = {
                task for task in self._runtime_tasks_stopping if not task.done()
            }
            if self._runtime_tasks_stopping:
                raise RuntimeError("plugin manager runtime tasks are still stopping")
            self._runtime_loop = loop
            pending = [state for state in self._background_task_factories.values()
                       if state.active and not state.scheduled]
        for state in pending:
            self._schedule_background_task(state, bind_context.copy())

    async def unbind_runtime_loop(
        self, loop: Optional[asyncio.AbstractEventLoop] = None, *,
        timeout: float = _RUNTIME_TASK_CANCEL_TIMEOUT_SECS,
    ) -> None:
        """Detach this profile manager and cancel its current task generation.

        Factory registrations remain active and pending so a later profile re-serve can bind the
        manager under a fresh runtime/secret scope. The state transition is lock-atomic, repeated
        calls after detach are inert, and a supplied non-owner loop is rejected. Cancellation is
        awaited asynchronously on the owner loop (including when called from another thread) so the
        gateway loop is never synchronously blocked.
        """
        with self._runtime_task_lock:
            bound_loop = self._runtime_loop
            if bound_loop is None:
                # A prior bounded call may still be waiting for cancellation-resistant Tasks.
                # Repeated unbinds join that same detached generation instead of falsely reporting
                # teardown complete and allowing profile deletion to proceed around it.
                tasks = [task for task in self._runtime_tasks_stopping if not task.done()]
                if not tasks:
                    return
                bound_loop = tasks[0].get_loop()
                if loop is not None and loop is not bound_loop:
                    raise RuntimeError("plugin manager is bound to a different runtime loop")
            else:
                if loop is not None and loop is not bound_loop:
                    raise RuntimeError("plugin manager is bound to a different runtime loop")
                self._runtime_loop = None
                tasks = []
                for state in self._background_task_factories.values():
                    if not state.active:
                        continue
                    # Invalidate both an already-created Task and a call_soon_threadsafe(_create)
                    # callback that has not run yet. The callback re-checks the owner loop below.
                    state.scheduled = False
                    task = state.task
                    state.task = None
                    if task is not None and not task.done():
                        tasks.append(task)
                        self._runtime_tasks_stopping.add(task)
        if not tasks:
            return

        async def _cancel_and_wait() -> set[asyncio.Task]:
            for task in tasks:
                task.cancel()
            _done, pending = await asyncio.wait(tasks, timeout=max(0.0, float(timeout)))
            return pending

        try:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - async method always has a caller loop
                current_loop = None
            if current_loop is bound_loop:
                pending = await _cancel_and_wait()
            else:
                cancellation = asyncio.run_coroutine_threadsafe(_cancel_and_wait(), bound_loop)
                pending = await asyncio.wrap_future(cancellation)
        finally:
            with self._runtime_task_lock:
                self._runtime_tasks_stopping = {
                    task for task in self._runtime_tasks_stopping if not task.done()
                }
        if pending:
            message = (
                f"Timed out after {timeout:.1f}s cancelling {len(pending)} plugin runtime "
                "background task(s); profile teardown cannot finish until they stop"
            )
            logger.error(message)
            raise TimeoutError(message)

    def _register_background_task(
        self, manifest: PluginManifest, name: str, factory: Callable[[], Any],
        registration_context: contextvars.Context,
    ) -> PluginRegistration:
        """Retain one owner-scoped factory and start it now when the runtime loop is already bound."""
        plugin_key = manifest_key(manifest)
        key = (plugin_key, name)
        state = _RegisteredBackgroundTask(plugin_key, name, factory, registration_context)
        with self._runtime_task_lock:
            incumbent = self._background_task_factories.get(key)
            if incumbent is not None and incumbent.active:
                raise ValueError(
                    f"Plugin {plugin_key!r} already registered background task {name!r}"
                )
            self._background_task_factories[key] = state

            def _release() -> None:
                self._release_background_task(key, state)

            try:
                handle = self._track_registration(
                    manifest, "background_task_factory", name, _release
                )
                state.handle = handle
            except BaseException:
                self._background_task_factories.pop(key, None)
                state.active = False
                raise
            loop_bound = self._runtime_loop is not None
        if loop_bound:
            try:
                self._schedule_background_task(state, registration_context.copy())
            except BaseException:
                handle.dispose()
                raise
        logger.debug("Plugin %s registered runtime background task: %s", plugin_key, name)
        return handle

    def _schedule_background_task(
        self, state: _RegisteredBackgroundTask, task_context: contextvars.Context,
    ) -> None:
        """Create ``state``'s Task on the bound loop, hopping threads without invoking its factory."""
        with self._runtime_task_lock:
            loop = self._runtime_loop
            if loop is None:
                return
            if loop.is_closed() or not loop.is_running():
                raise RuntimeError("registered background task cannot use a stopped host asyncio loop")
            if not state.active or state.scheduled:
                return
            state.scheduled = True

        def _create() -> None:
            with self._runtime_task_lock:
                if (self._runtime_loop is not loop or not state.active
                        or not state.scheduled or state.task is not None):
                    return
                if loop.is_closed() or not loop.is_running():
                    state.scheduled = False
                    raise RuntimeError(
                        f"Plugin {state.plugin_key!r} background task {state.name!r} lost its running host loop"
                    )
                runner = self._run_background_factory(state)
                try:
                    task = loop.create_task(
                        runner,
                        name=f"plugin:{state.plugin_key}:{state.name}",
                        context=task_context.copy(),
                    )
                except BaseException:
                    runner.close()
                    state.scheduled = False
                    raise
                state.task = task
            task.add_done_callback(lambda completed: self._background_task_done(state, completed))

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        try:
            if current_loop is loop:
                _create()
            else:
                loop.call_soon_threadsafe(_create, context=task_context)
        except BaseException:
            with self._runtime_task_lock:
                state.scheduled = False
            raise

    @staticmethod
    async def _run_background_factory(state: _RegisteredBackgroundTask) -> Any:
        produced = state.factory()
        if not inspect.iscoroutine(produced):
            raise TypeError("background task factory must return a coroutine")
        return await produced

    def _background_task_done(
        self, state: _RegisteredBackgroundTask, completed: asyncio.Task,
    ) -> None:
        with self._runtime_task_lock:
            self._runtime_tasks_stopping.discard(completed)
            current_generation = state.task is completed
        # Unbind detached this generation and deliberately retained its registration. Its
        # cancellation callback must not dispose the factory needed by a later re-serve.
        if not current_generation:
            return
        try:
            if not completed.cancelled() and (exc := completed.exception()) is not None:
                logger.error(
                    "Plugin %s background task %s failed: %s",
                    state.plugin_key, state.name, exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        finally:
            handle = state.handle
            if handle is not None:
                handle.dispose()

    def _release_background_task(
        self, key: Tuple[str, str], state: _RegisteredBackgroundTask,
    ) -> None:
        with self._runtime_task_lock:
            if not state.active:
                return
            state.active = False
            if self._background_task_factories.get(key) is state:
                self._background_task_factories.pop(key, None)
            task = state.task
            loop = self._runtime_loop
        if task is None or task.done():
            return
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            task.cancel()
        elif loop is not None:
            loop.call_soon_threadsafe(task.cancel)
