"""Tests for the replay band, scripted user, fake app and frame-timing maths (no window, no model)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app.fake import FakeEmailApp  # noqa: E402
from ui.session import ScriptedUser, Step  # noqa: E402
from ui.strip import frame_report, luminance  # noqa: E402

DATA = ROOT / "data"
MENU = [9.25, 10.25, 11.25, 12.25, 13.25, 14.25]


def test_luminance_example():
    assert luminance([10.0], [0.0], 3, 60)[0] == pytest.approx(0.5)  # sin(pi) = 0
    assert luminance([10.0], [0.0], 1.5, 60)[0] == pytest.approx(1.0)  # sin(pi/2) = 1


def test_frame_report_counts_dropped_frames():
    t = np.arange(600) / 60.0
    log = [(k, k) for k in range(600)]
    assert frame_report(t, log, [10.0], [0.0], 60.0)["dropped_frames"] == 0
    t2 = np.concatenate([t[:300], t[300:] + 1 / 60.0])  # one refresh missed
    r = frame_report(t2, log, [10.0], [0.0], 60.0)
    assert r["dropped_frames"] == 1 and r["dropped_pct"] == pytest.approx(100 / 599)


@pytest.mark.skipif(not (DATA / "wearable" / "S028.mat").exists(), reason="D4 data not present")
def test_replay_band_streams_requested_trials():
    from band.replay import ReplayBand
    from decoder.datasets import NAKANISHI_FREQS
    b = ReplayBand(DATA, 28, ["OZ", "POZ", "O1", "O2", "PO5", "PO6"], MENU, seed=0, realtime=False)
    b.start()
    b.read()
    t_on = b.look(10.25)
    _, f_req, k, f_true = b.log[-1]
    assert f_req == 10.25 and f_true == 10.25
    n0 = round(t_on * b.fs) - b.onset  # first sample of the trial in the stream
    got = []
    while sum(c.shape[1] for c in got) < 700:
        c, t_end = b.read()
        if round(t_end * b.fs) > n0:
            got.append(c)
    stream = np.concatenate(got, 1)
    assert np.allclose(stream[:, :600], b.X[k][:, :600])
    from band.replay import _Pool
    pool = _Pool(b.idle_pool.idx, np.random.default_rng(1))
    idle = [pool.draw() for _ in range(60)]
    assert not set(NAKANISHI_FREQS[b.y[idle]]) & set(MENU)  # idle trials are never menu frequencies
    assert len(set(idle)) == 60  # no repeats before the pool is used up
    b.stop()


def test_scripted_user_recovers_and_idles():
    app = FakeEmailApp()
    steps = [Step(("home", None), "Email"), Step(("inbox", None), app.inbox_labels()[0]),
             Step(("email", 0), "Reply", idle_before=2)]
    u = ScriptedUser(steps)
    lab = lambda: [o.label for o in app.menu().options]  # noqa: E731
    assert u.target(app.state(), lab(), False) == "Email"
    app.pick(app.menu().options[0])
    app.pick(app.menu().options[1])  # wrong e-mail opened
    assert u.target(app.state(), lab(), False) == "Back"
    app.pick([o for o in app.menu().options if o.label == "Back"][0])
    app.pick(app.menu().options[0])  # the right e-mail
    assert [u.target(app.state(), lab(), False) for _ in range(3)] == [None, None, "Reply"]
    assert u.target(app.state(), lab(), True) == "Resume"


def test_fake_app_archiving_the_target_fails_the_task():
    app = FakeEmailApp(target_email=0)
    app.pick(app.menu().options[0])
    app.pick(app.menu().options[0])
    app.pick([o for o in app.menu().options if o.label == "Archive"][0])
    assert app.done and app.success is False


def test_scripted_user_binds_top_item_and_finds_it_again():
    u = ScriptedUser([Step(("inbox", None), "#0", bind="email"), Step(("email", "$email"), "Reply")])
    assert u.target(("inbox", None), ["Lunch", "Report", "Back"], False) == "Lunch"
    assert u.bound["email"] == "Lunch"
    assert u.target(("email", "Report"), ["Reply", "Archive", "Back"], False) == "Back"  # wrong e-mail open
    assert u.target(("inbox", None), ["Report", "Lunch", "Back"], False) == "Lunch"  # order changed: still Lunch
    assert u.target(("email", "Lunch"), ["Reply", "Archive", "Back"], False) == "Reply"
