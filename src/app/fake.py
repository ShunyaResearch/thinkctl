"""FakeEmailApp: the email flow's menus and transitions without the model or the screen (wiring test).
Same interface as the real app (app/email_app.py): menu(), state(), pick(option), busy, paused, done, success, status.
"""
import time

from agent import options as O

EMAILS = ["Lunch on Thursday?", "Draft report review", "Reunión del viernes", "Conference travel"]
DRAFTS = ["Yes, see you then.", "Sorry, I can't make it.", "Which place?", "Let me check later."]


class FakeEmailApp:
    display_name = "fake app (no Gmail)"
    def __init__(self, target_email=0, target_draft=1, delay_s=0.0):
        self.screen, self.ctx = "home", None
        self.paused, self.busy, self.done, self.success = False, False, False, None
        self.target = (target_email, target_draft)
        self.delay_s, self.status, self.history = delay_s, "", []
        self.archived = set()

    def state(self):
        return (self.screen, self.ctx)

    def _inbox(self):
        return [(k, O.short(e)) for k, e in enumerate(EMAILS) if k not in self.archived][:4]

    def inbox_labels(self):
        return [l for _, l in self._inbox()]

    def menu(self):
        s = self.screen
        if s == "home":
            return O.home()
        if s == "inbox":
            opts = [O.Option(l, "goal", None, {"email": k}) for k, l in self._inbox()]
            return O.Menu("inbox", opts + [O.Option("Back", "back"), O.PAUSE])
        if s == "email":
            return O.email_menu({})
        if s == "drafts":
            return O.Menu("drafts", [O.Option(O.short(d), "goal", None, {"draft": j}) for j, d in enumerate(DRAFTS)] +
                          [O.Option("Back", "back"), O.PAUSE])
        if s == "typed":
            return O.typed_menu()
        if s == "confirm":
            return O.confirm_menu()
        raise ValueError(s)

    def pick(self, opt):
        self.history.append((self.screen, self.ctx, getattr(opt, "label", opt)))
        if self.delay_s:
            time.sleep(self.delay_s)
        if opt == "resume":
            self.paused = False
            return
        if opt is None or opt.kind == "noop":
            return
        if opt.kind == "pause":
            self.paused = True
            return
        s, lab = self.screen, opt.label
        if opt.kind == "back":
            if s == "inbox":
                self.screen, self.ctx = "home", None
            elif s == "email":
                self.screen, self.ctx = "inbox", None
            elif s == "drafts":
                self.screen = "email"
            elif s == "typed":
                self.screen, self.ctx = "drafts", self.ctx[0]
            elif s == "confirm":
                self.screen = "typed"
            return
        if s == "home" and lab == "Email":
            self.screen, self.ctx = "inbox", None
        elif s == "inbox":
            self.screen, self.ctx = "email", opt.data["email"]
        elif s == "email" and lab == "Reply":
            self.screen = "drafts"
        elif s == "email" and lab == "Archive":
            self.archived.add(self.ctx)
            if self.ctx == self.target[0]:
                self.done, self.success, self.status = True, False, "target e-mail archived by a wrong pick"
            self.screen, self.ctx = "inbox", None
        elif s == "drafts":
            self.screen, self.ctx = "typed", (self.ctx, opt.data["draft"])
        elif s == "typed" and lab == "Send":
            self.screen = "confirm"
        elif s == "typed" and lab == "Edit-cancel":
            self.screen, self.ctx = "email", self.ctx[0]
        elif s == "confirm" and lab == "Confirm send":
            self.done = True
            self.success = self.ctx == self.target
            self.status = f"sent e-mail {self.ctx[0]} draft {self.ctx[1]}"
