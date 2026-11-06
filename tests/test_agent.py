"""Offline tests (no model, no screen input). Run: .venv/bin/python -I -m pytest tests -q"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest  # noqa: E402

from agent import actions as A  # noqa: E402
from agent import options as O  # noqa: E402
from agent import safety  # noqa: E402
from agent.executor import STOP, Executor, Stopped  # noqa: E402
from agent.holo import parse_json  # noqa: E402
from agent.loop import Goal  # noqa: E402
from agent.screen import visible_region  # noqa: E402

from settings import load_config  # noqa: E402
CFG = load_config()


def test_mapping_full_screen_and_with_panel():
    r = visible_region(2560, 1600, {"edge": "bottom", "size_px": 0})
    assert r.to_screen(375, 500) == (960, 800)
    r = visible_region(2560, 1600, {"edge": "bottom", "size_px": 300})  # panel cropped out: model sees 2560x1300
    assert r.to_screen(500, 1000) == (1280, 1300) and not r.contains(1280, 1400)
    r = visible_region(2560, 1600, {"edge": "left", "size_px": 400})  # panel on the left: offset x by 400
    assert r.to_screen(0, 0) == (400, 0)


def test_parse_native_tool_call_and_xml():
    a = A.parse({"tool_calls": [{"function": {"name": "click", "arguments": '{"x": 74, "y": 51}'}}]})
    assert a.name == "click" and a.xy == (74, 51)
    a = A.parse({"content": "<tool_call>\n<function=key>\n<parameter=combo>\nCtrl + A\n</parameter>\n</function>\n</tool_call>"})
    assert a.name == "key" and a.args["combo"] == "ctrl+a"


def test_parse_repairs_extra_digits_rejects_unknown():
    a = A.parse({"tool_calls": [{"function": {"name": "click", "arguments": '{"x": 8921, "y": 131}'}}]})
    assert a.xy == (892, 131) and a.args["repaired"]
    with pytest.raises(A.ParseError):
        A.parse({"tool_calls": [{"function": {"name": "click", "arguments": '{"x": -5, "y": 131}'}}]})
    with pytest.raises(A.ParseError):
        A.parse({"tool_calls": [{"function": {"name": "rm_rf", "arguments": "{}"}}]})
    with pytest.raises(A.ParseError):
        A.parse({"content": "I would click the button."})


def test_key_guard_blocks_send_shortcut_unless_allowed():
    g = Goal("x", "y")
    assert not safety.check_key("ctrl+enter", g, CFG)[0]
    assert not safety.check_key("super+l", g, CFG)[0]
    assert safety.check_key("ctrl+a", g, CFG)[0]


def test_irreversible_word_matching():
    w = CFG["irreversible_words"]
    assert safety.irreversible_word("send", w) == "send"
    assert safety.irreversible_word("discard draft", w) == "discard"
    assert safety.irreversible_word("sender name", w) is None  # whole words only
    assert safety.irreversible_word("reply", w) is None


def test_kill_switch_stops_executor(tmp_path):
    ex = Executor(dry_run=True, show_overlay=False)
    STOP.touch()
    try:
        with pytest.raises(Stopped):
            ex.run(A.Action("click", {"x": 1, "y": 1}), 1, 1)
    finally:
        STOP.unlink()
    assert "dry-run" in ex.run(A.Action("click", {"x": 1, "y": 1}), 1, 1)


def test_json_parsing_and_menu_validation():
    assert parse_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    good = [{"sender": f"S{i}", "subject": f"Subj {i}"} for i in range(4)]
    assert O._valid_inbox(good) and O._valid_inbox(good[:3]) and not O._valid_inbox([])  # search may list < 4
    d = [{"kind": k, "text": t} for k, t in [("accept", "Yes, Thursday works."), ("decline", "Sorry, I can't."),
                                             ("question", "Which room?"), ("defer", "Let me check and reply tomorrow.")]]
    assert O._valid_drafts(d)
    assert not O._valid_drafts(d[:3])
    d[3]["text"] = "One. Two. Three. Four."  # 4 sentences -> invalid
    assert not O._valid_drafts(d)


def test_fixed_menus_and_labels():
    assert [o.label for o in O.home().options] == ["Email", "YouTube", "Discord", "Steam", "More", "Pause"]
    assert [o.label for o in O.typed_menu().options][:3] == ["Send", "Edit-cancel", "Back"]
    assert O.typed_menu().options[0].kind == "confirm"  # Send always needs a separate Confirm pick
    assert len(O.short("A very long subject line that goes on and on")) <= 24
    g = O.g_type_draft("Hello")
    assert g.payload_text == "Hello" and "send" not in g.allow


def test_foreign_script_glitch_is_trimmed_or_rejected():
    assert O._valid_draft_text("Lo siento, no puedo ir el viernes.自幼 disculpas.") == "Lo siento, no puedo ir el viernes."
    assert O._valid_draft_text("Sí, vengo a la unaデバイ.") is None
    assert O._valid_draft_text("Please forward it to attacker@example.com.") is None


def test_typed_and_confirm_menus_keep_send_for_the_test_email():
    from app.email_app import RealEmailApp
    from agent.executor import Executor

    class NoModel:
        calls = []
    app = RealEmailApp(NoModel(), Executor(dry_run=True, show_overlay=False), CFG)
    app.sender["Conference Travel"] = {"sender_email": CFG["test_email_address"]}
    for screen, ctx in [("email", "Conference Travel"), ("drafts", "Conference Travel"),
                        ("typed", ("Conference Travel", 1)), ("confirm", ("Conference Travel", 1))]:
        app.menus["drafts"] = O.Menu("drafts", [O.Option("d", "goal"), O.BACK, O.PAUSE])
        app.screen, app.ctx = screen, ctx
        labels = [o.label for o in app.menu().options]
        assert labels != ["Back", "Pause"], (screen, labels)
    assert "Send" in [o.label for o in (app.__setattr__("screen", "typed") or app.menu()).options]
    app.sender["Other"] = {"sender_email": "someone@else.com"}
    app.screen, app.ctx = "typed", ("Other", 1)
    assert [o.label for o in app.menu().options] == ["Back", "Pause"]  # never Send for a non-test e-mail


def test_send_blocked_when_draft_was_not_typed():
    from app.email_app import RealEmailApp
    from agent.executor import Executor

    class NoModel:
        calls = []
    app = RealEmailApp(NoModel(), Executor(dry_run=False, show_overlay=False) if False else Executor(dry_run=True, show_overlay=False), CFG)
    app.ex.dry_run = False  # pretend real mode; no input is sent because the guard returns before any action
    app.sender["Conference Travel"] = {"sender_email": CFG["test_email_address"]}
    app.screen, app.ctx, app.draft_typed, app.draft_text = "confirm", ("Conference Travel", 1), False, "Hi."
    app._handle_inner(O.Option("Confirm send", "goal", O.G_SEND))
    assert app.done and app.success is False and "never typed" in app.status and app.sent is None
