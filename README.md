# thinkctl

Control a computer with your eyes on a flickering menu: a steady-state visual evoked potential (SSVEP)
brain-computer interface picks options, and a local computer-use model (Holo3.1-4B) carries them out on screen.
Everything runs locally; the GPU is used only by the model.

This is a research prototype. It currently runs on **recorded** EEG from public datasets (replayed in real time as if
from a headband), not on live EEG.

## What is here
- **Decoder** (`src/decoder/`, CPU): FBCCA, transfer templates (zero calibration), ensemble TRCA, idle detection,
  dynamic stopping, and a real-time stream decoder.
- **Offline experiments** (`src/experiments/`): the experiments that chose the decoder's defaults on four public
  datasets.
- **Replay demo** (`src/demo.py`, `src/band/`, `src/ui/`, `src/agent/`, `src/app/`): EEG replay → decoder → 6-button
  flicker strip → menus → Holo3.1-4B agent that reads and replies to e-mail in Gmail, with safety checks in code.

## Layout
```
src/
  decoder/      SSVEP decoding library (CPU)
  experiments/  offline experiments E1–E8
  band/         EEG replay, decoder process, decoder model building / tuning
  ui/           flicker strip and session loop
  agent/        computer-use loop, option generator, safety checks
  app/          e-mail app (real Gmail) and a model-free stand-in
  scripts/      data / model / llama.cpp setup, model server, kill switch, tools
  demo.py       one end-to-end session
docs/           setup.md, decoder.md, demo.md
tests/          offline tests
```

## Quick start
```bash
uv venv .venv --python 3.10 && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -I -m pytest tests -q
```
Then follow [docs/setup.md](docs/setup.md) (datasets, llama.cpp, model, decoder files, config),
[docs/decoder.md](docs/decoder.md) (offline experiments) and [docs/demo.md](docs/demo.md) (running the demo).

## Results so far

All decoder numbers are offline, on public data, with every tuned choice made on one half of the subjects and
reported on the other half (2-fold). D3 (recorded outside the lab) and D4 (8-channel headband) are treated as the
realistic datasets; D1/D2 (lab) are optimistic. Menus have 6 options.

### Decoder (offline, held-out subjects)
| Setting (6 options) | Accuracy | Users ≥ 90 % | ITR |
|---|---|---|---|
| D4 wet, transfer templates, no calibration, 1 s / 2 s | 84 % / 93 % | 47 % / 75 % | 69 / 51 bits/min |
| D4 wet, transfer templates, dynamic stopping (mean 1.0 s) | 90 % | 70 % | 87 bits/min |
| D4 wet, ensemble TRCA (9 calibration blocks), 1 s | 91 % | 73 % | 83 bits/min |
| D4 dry, best method (FBCCA), 2 s | 76 % | 35 % | 35 bits/min |

- Channels: 6 parieto-occipital channels (Oz, POz, O1, O2, PO5, PO6); 4 cost 2–5 points (D1/D2).
- Zero calibration: averaging other users' responses (transfer templates) beats plain FBCCA by about 3–4 points (D2/D3).
- Dynamic stopping (stop as soon as confident) gives the same accuracy in about a third less time: +35–53 % ITR (D3, D4 wet).
- "Not looking" detection at ≤ 1 false activation per idle minute, 2 s window, transfer templates: 65–73 % of stares
  accepted at the first attempt, 99 % of accepted picks correct (D3, D4 wet).
- Dry electrodes: 14–19 points below wet with FBCCA; template methods do not transfer to dry recordings.

### Real-time pipeline (replay of recorded EEG)
- Stream decoder vs offline pipeline: 99.3–100 % identical decisions (D2, 34 subjects, 2,448 attempts per setting);
  2–5 ms CPU per decision on one laptop core; 64 ms from window end to decision (mostly the 0.1 s data chunks).
- Demo menu (9.25–14.25 Hz, 6 channels, zero calibration, D4 wet, 102 subjects): 79 % at 1 s, 90 % at 2 s.
  Combining evidence over up to 3 attempts: correct pick within 4 attempts 70 % → 83 %, users stuck 28 % → 16 %,
  false activations 1.0 → 0.85 per idle minute. Swapping to a second frequency set when inconclusive: neutral
  on average, helps the weakest users slightly.
- Flicker strip (sampled sinusoidal stimulation, 60 Hz display): 0 dropped frames in 12,570 flicker frames; achieved
  frequencies within ±0.002 Hz of target.
- Replay sessions of the full menu flow (6 picks, 5 randomly drawn held-out subjects): 4 of 5 completed in 23–55 s
  with 0 wrong picks; 1 subject never responds at 9.25 Hz and stays stuck.

### Computer-use agent (Holo3.1-4B, Q4, local llama.cpp, 8 GB laptop GPU)
- 4.3 GB GPU memory; about 1.3–5 s per action step (screenshot to action), 0.3–0.5 s per safety check.
- Click grounding on desktop screenshots: 6/8 correct (7/8 with model "thinking", about 9 s per call). On recorded
  Gmail screens: 14/15 planned actions correct; 3/3 safety-guard tests blocked (send without confirm, delete,
  misplaced discard); an e-mail containing a prompt-injection did not change the agent's goals or the drafted replies.
- End to end with real clicks: replayed EEG picked Email → e-mail → Reply → draft → Send → Confirm, and the reply
  (in the e-mail's language, Spanish) was sent to the test address in 60 s (EEG 18 s, menu generation 8 s, agent 26 s).
- Lesson from testing: the model's own "did it work?" answers were wrong several times (an unsent search, an empty
  reply box), so every critical step (which e-mails are listed, what is typed, whether Send may happen) is checked in
  code. 31 offline tests cover the decoder, the stream decoder, the agent's parsing and guards, and the UI logic.

### Not done yet
- Live EEG from a headband (all results are offline or replayed recordings).
- A larger end-to-end evaluation (only one full real run so far) and screen recordings of runs.
- Navigation on the e-mail screens (choose any e-mail, scroll); other apps behind the home menu.
- Running the decoder on a microcontroller (estimated feasible on an ESP32-S3 from FLOP counts, not measured).

## References

### Datasets
- D1. Nakanishi M, Wang Y, Wang Y-T, Jung T-P. A comparison study of canonical correlation analysis based methods for
  detecting steady-state visual evoked potentials. *PLoS ONE* 10(10):e0140703, 2015.
  Data: https://github.com/mnakanishi/12JFPM_SSVEP
- D2. Wang Y, Chen X, Gao X, Gao S. A benchmark dataset for SSVEP-based brain–computer interfaces.
  *IEEE Trans Neural Syst Rehabil Eng* 25(10):1746–1752, 2017. Data mirror: NEMAR nm000126.
- D3. Liu B, Huang X, Wang Y, Chen X, Gao X. BETA: a large benchmark database toward SSVEP-BCI application.
  *Front Neurosci* 14:627, 2020. Data mirror: https://huggingface.co/datasets/Bingchuan/BETA
- D4. Zhu F, Jiang L, Dong G, Gao X, Wang Y. An open dataset for wearable SSVEP-based brain-computer interfaces.
  *Sensors* 21(4):1256, 2021. Data: https://doi.org/10.6084/m9.figshare.13560281

### Methods
- Lin Z, Zhang C, Wu W, Gao X. Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs.
  *IEEE Trans Biomed Eng* 53(12):2610–2614, 2006.
- Chen X, Wang Y, Gao S, Jung T-P, Gao X. Filter bank canonical correlation analysis for implementing a high-speed
  SSVEP-based brain–computer interface. *J Neural Eng* 12(4):046008, 2015.
- Chen X, Wang Y, Nakanishi M, Gao X, Jung T-P, Gao S. High-speed spelling with a noninvasive brain–computer
  interface. *PNAS* 112(44):E6058–E6067, 2015.
- Nakanishi M, Wang Y, Chen X, Wang Y-T, Gao X, Jung T-P. Enhancing detection of SSVEPs for a high-speed brain
  speller using task-related component analysis. *IEEE Trans Biomed Eng* 65(1):104–112, 2018.
- Yuan P, Chen X, Wang Y, Gao X, Gao S. Enhancing performances of SSVEP-based brain–computer interfaces via
  exploiting inter-subject information. *J Neural Eng* 12(4):046006, 2015.
- Manyakov NV, Chumerin N, Robben A, Combaz A, van Vliet M, Van Hulle MM. Sampled sinusoidal stimulation profile and
  multichannel fuzzy logic classification for monitor-based phase-coded SSVEP brain–computer interfacing.
  *J Neural Eng* 10(3):036011, 2013.
- Wolpaw JR, Birbaumer N, McFarland DJ, Pfurtscheller G, Vaughan TM. Brain–computer interfaces for communication and
  control. *Clin Neurophysiol* 113(6):767–791, 2002.

### Model and tools
- H Company. Holo3.1: fast and local computer use agents, 2026. https://huggingface.co/Hcompany/Holo-3.1-4B
  (Apache-2.0; based on the Qwen3.5 family).
- llama.cpp / ggml. https://github.com/ggml-org/llama.cpp

## License
Author: Sachit Ramesha Gowda. MIT (see LICENSE). Datasets and the model have their own licences (CC-BY-4.0 for D2–D4, Apache-2.0 for Holo3.1).
