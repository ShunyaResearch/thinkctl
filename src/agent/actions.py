"""The action space offered to the model (native function calling) and strict parsing / validation."""
import json, re
from dataclasses import dataclass, field

_xy = {"x": {"type": "integer", "description": "0-1000, left to right"},
       "y": {"type": "integer", "description": "0-1000, top to bottom"}}


def _tool(name, desc, props=None, req=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": list(req)}}}


TOOLS = [
    _tool("click", "Left-click a point on the screenshot.", _xy, ["x", "y"]),
    _tool("double_click", "Double-click a point on the screenshot.", _xy, ["x", "y"]),
    _tool("type_text", "Type the goal's text into the focused text field. The text itself is supplied by the system."),
    _tool("key", "Press a key or key combination, e.g. 'Escape', 'ctrl+a'.",
          {"combo": {"type": "string"}}, ["combo"]),
    _tool("scroll", "Scroll at a point.", {**_xy, "direction": {"type": "string", "enum": ["up", "down"]},
                                           "amount": {"type": "integer", "description": "1-10 wheel steps"}},
          ["x", "y", "direction"]),
    _tool("done", "The goal is achieved (state what you see that proves it).", {"evidence": {"type": "string"}},
          ["evidence"]),
    _tool("fail", "The goal cannot be achieved from here.", {"reason": {"type": "string"}}, ["reason"]),
]
NAMES = {t["function"]["name"] for t in TOOLS}
POINT = {"click", "double_click", "scroll"}


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)
    raw: str = ""

    @property
    def xy(self):
        return (self.args["x"], self.args["y"]) if self.name in POINT else None

    def short(self):
        a = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.name}({a})"


class ParseError(ValueError):
    pass


def parse(msg):
    """llama-server message -> Action. Accepts parsed tool_calls or the raw <tool_call> XML in the text."""
    calls = msg.get("tool_calls") or []
    if calls:
        f = calls[0]["function"]
        args = f.get("arguments") or {}
        args = json.loads(args) if isinstance(args, str) else args
        return validate(Action(f["name"], args, raw=json.dumps(f)))
    text = msg.get("content") or ""
    m = re.search(r"<function=(\w+)>(.*?)</function>", text, re.S)
    if m:
        args = {k: v.strip() for k, v in re.findall(r"<parameter=(\w+)>(.*?)</parameter>", m.group(2), re.S)}
        return validate(Action(m.group(1), args, raw=m.group(0)))
    raise ParseError(f"no function call in model output: {text[-200:]!r}")


def validate(a):
    if a.name not in NAMES:
        raise ParseError(f"unknown action {a.name}")
    if a.name in POINT:
        try:
            a.args["x"], a.args["y"] = int(float(a.args["x"])), int(float(a.args["y"]))
        except (KeyError, ValueError, TypeError) as e:
            raise ParseError(f"bad coordinates {a.args}") from e
        for k in ("x", "y"):  # Holo sometimes appends digits (2641 for 264, 331569 for 331): keep the first 3
            v = a.args[k]
            if v > 1000 and str(v)[0] != "0":
                a.args[k] = int(str(v)[:3])
                a.args["repaired"] = True
        if not (0 <= a.args["x"] <= 1000 and 0 <= a.args["y"] <= 1000):
            raise ParseError(f"coordinates outside 0-1000: {a.args}")
    if a.name == "scroll":
        a.args["amount"] = max(1, min(10, int(float(a.args.get("amount", 3)))))
        if a.args.get("direction") not in ("up", "down"):
            raise ParseError("scroll direction must be up/down")
    if a.name == "key":
        combo = str(a.args.get("combo", "")).strip().lower().replace(" ", "")
        if not re.fullmatch(r"([a-z0-9_]+\+)*[a-z0-9_]+", combo):
            raise ParseError(f"bad key combo {combo!r}")
        a.args["combo"] = combo
    return a
