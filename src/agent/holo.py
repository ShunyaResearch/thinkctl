"""Minimal client for the local Holo3.1-4B server (llama-server, OpenAI-compatible, 127.0.0.1 only)."""
import base64, io, json, re, time, urllib.request


def b64png(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Holo:
    def __init__(self, url, timeout=180):
        if not url.startswith("http://127.0.0.1"):
            raise ValueError("the model server must be local (127.0.0.1)")
        self.url, self.timeout = url, timeout
        self.calls = []  # (purpose, seconds, prompt_tokens, completion_tokens)

    def chat(self, system, user_text, images=(), tools=None, think=False, max_tokens=None, purpose="", temperature=0.0):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64png(im)}} for im in images]
        content.append({"type": "text", "text": user_text})
        body = {"model": "holo", "temperature": temperature, "max_tokens": max_tokens or (2048 if think else 512),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "chat_template_kwargs": {"enable_thinking": think}}
        if not think:
            body["reasoning_effort"] = "none"
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        r = json.load(urllib.request.urlopen(req, timeout=self.timeout))
        dt = time.perf_counter() - t0
        u = r.get("usage", {})
        self.calls.append((purpose, dt, u.get("prompt_tokens"), u.get("completion_tokens")))
        msg = r["choices"][0]["message"]
        msg["_seconds"] = dt
        return msg

    def json(self, system, user_text, images=(), think=False, purpose="", temperature=0.0):
        """Ask for JSON; returns (parsed or None, raw text, seconds)."""
        msg = self.chat(system, user_text, images, think=think, purpose=purpose, max_tokens=2048 if think else 1024,
                        temperature=temperature)
        text = (msg.get("content") or "").strip()
        return parse_json(text), text, msg["_seconds"]


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    for cand in (text, _outer(text, "[", "]"), _outer(text, "{", "}")):
        if cand:
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                pass
    objs = []  # tolerate a list written without brackets: {...}{...} or {...},{...}
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            objs.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            return None
    return objs or None


def _outer(text, a, b):
    i, j = text.find(a), text.rfind(b)
    return text[i:j + 1] if 0 <= i < j else None
