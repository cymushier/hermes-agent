"""A one-shot ``chat -q`` query is echoed literally, never parsed as Rich markup.

Autonomy and swarm workers pass prompts full of brackets (``[/path]``, ``[bold]``, closing
tags). Interpolated raw into the label, Rich raised ``MarkupError`` before the agent ran.
"""

from __future__ import annotations

from types import SimpleNamespace

from rich.console import Console

import cli


def test_query_label_is_escaped_not_parsed_as_markup(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(cli, "_should_seed_interactive", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_collect_query_images", lambda q, i: (q, []))
    monkeypatch.setattr(cli, "_collect_kanban_task_images", lambda imgs: [])
    monkeypatch.setattr(cli, "_finalize_single_query", lambda c: None)
    console = Console(record=True, width=200, color_system=None)
    stub = SimpleNamespace(
        _single_query_mode=False,
        _claim_active_session=lambda *a, **k: True,
        console=console,
        _show_security_advisories=lambda: None,
        chat=lambda *a, **k: "response",
        _print_exit_summary=lambda **k: None,
        _last_turn_result={"final_response": "done", "completed": True},
    )
    query = "fix [/tmp/x] then [bold]keep[/bold] the [/] literal"
    try:
        cli._run_single_query_mode(stub, query, None, False, True)
    except SystemExit as exc:
        assert exc.code in (0, None)
    assert query in console.export_text()
