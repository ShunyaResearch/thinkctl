# Replay demo: EEG → flicker strip → computer-use agent

Recorded EEG of a held-out subject (Wearable dataset, wet electrodes) is streamed in real time as if from a headband.
A scripted user decides where the recorded subject "looks"; the decoder turns the EEG into picks on a 6-button flicker
strip; picks drive menus and a local Holo3.1-4B agent that operates Gmail in Firefox.

**Photosensitivity warning:** the strip flickers at 9–15 Hz. A warning is shown on first start.

## Run
Inside `nix develop path:.`, with setup steps 1–7 done (see `setup.md`):
```bash
bash src/scripts/run_holo_server.sh &      # model server on 127.0.0.1:8090; refuses if the GPU is busy
bash src/scripts/killswitch.sh &           # Ctrl+Alt+K creates the STOP file: no input, flicker stops
.venv/bin/python -I src/demo.py --app fake  --subject 32 --seed 2              # menus only, no model
.venv/bin/python -I src/demo.py --app email --subject 28 --seed 2 --dry-run    # real Gmail, actions only drawn
.venv/bin/python -I src/demo.py --app email --subject 28 --seed 2              # real Gmail, real clicks and send
```
Replay subjects: 28, 32, 52, 64, 84 (drawn at random with a fixed seed). Remove `STOP` to re-arm after the kill switch.
Do not use the mouse or keyboard during a real run.

## Flow
Home (Email, YouTube, Discord, Steam, More, Pause) → the 4 latest e-mails from the test sender → Reply / Archive / Back
→ 4 model-written replies → Send / Edit-cancel / Back → Confirm → sent. YouTube, Discord, Steam and More do nothing yet.

## How a pick works
- Attempt: 0.5 s cue, then flicker; the decoder decides after 0.5–2.0 s (dynamic stopping) or reports no pick.
- Evidence is combined over up to 3 attempts on the same screen; after K inconclusive attempts the strip swaps to
  the alternate frequency set. Pause mode shows only Resume and uses a stricter threshold.
- The strip shows the decoder state, a confidence bar, per-button evidence and (replay only) the simulated gaze.

## Safety
- Only e-mails from `test_email_address` are listed (the Gmail search is opened by code) and only those can be
  replied to; for any other sender only Back and Pause are offered.
- Before every click, a zoomed crop asks the model what is under the point; send, delete, forward and similar actions
  are blocked unless the goal explicitly allows them.
- Send needs a separate Confirm pick, happens at most once per draft, and only if the draft was typed by code and is
  visible in the reply box. Goals are fixed templates; text on the screen is treated as data.
- The model server listens on 127.0.0.1 only; screenshots never leave the machine.

## Tools
```bash
.venv/bin/python -I src/scripts/ui_frame_test.py --seconds 60   # frame timing: dropped frames, achieved frequency
.venv/bin/python -I src/scripts/summarize_runs.py                # table of logs/runs/
```

## Known issues
- `--record` (ffmpeg x11grab) currently produces no file.
- The model's own "did it work?" answers are unreliable; critical steps are therefore checked in code.
- Some users never respond to some frequencies (e.g. one replay subject at 9.25 Hz) and can get stuck.
- Planned: navigation options on the inbox and e-mail screens (choose any e-mail, scroll).
