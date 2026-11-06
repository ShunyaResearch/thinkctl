# Decoder and offline experiments

`src/decoder/` is a CPU-only SSVEP decoding library; `src/experiments/` contains the offline experiments that chose its
defaults. All experiments run on the public datasets in `data/` and write CSV/JSON to `results/` and figures to
`figures/` (both git-ignored).

## Default pipeline
- 50 Hz notch (Q = 30).
- Stare window starts 0.14 s after stimulus onset (visual latency); 0.5 s gaze shift per selection for ITR.
- Filter bank: Chebyshev-I, order 4, 0.5 dB ripple, zero-phase on the window; sub-band m covers [8m − 2, 88] Hz
  (6, 14, 22, 30, 38 Hz), weights w_m = m^−1.25 + 0.25; 5 harmonics.
- Methods: FBCCA (no calibration), transfer templates (average of other users, zero calibration), ensemble TRCA
  (user's own calibration).
- Confidence: margin = best score − second-best score; idle threshold for ≤ 1 false activation per idle minute.
- Every tuned choice is made on one half of the subjects and scored on the other half, then swapped (2-fold).

## Experiments
| Script | Question |
|---|---|
| `check_datasets.py` | labels, channel order and mains frequency of every dataset |
| `e1_fbcca_fix.py` | reproduce FBCCA vs CCA and tune the filter bank |
| `e2_channels.py` | how few channels are enough |
| `e3_options_vs_time.py` | number of options vs stare time (accuracy, ITR) |
| `e4_calibration.py` | how much calibration helps (FBCCA, transfer templates, TRCA) |
| `build_score_cache.py` | per-trial scores for 6-option menus, shared by E5–E7 |
| `e5_dry_vs_wet.py`, `e5_d4_checks.py` | dry vs wet electrodes; electrode order and mains handling in D4 |
| `e6_idle_detection.py` | "not looking" detection: true picks vs false activations |
| `e7_dynamic_stopping.py` | grow the window until confident vs a fixed window |
| `e8_streaming.py` | real-time stream decoder: equivalence, latency, CPU, memory, microcontroller estimate |
| `notch_check.py` | effect of the 50 Hz notch |

```bash
.venv/bin/python -I src/experiments/e7_dynamic_stopping.py --data data/ --out results/ --figs figures/
```

## Real-time decoder
`src/decoder/stream_decoder.py`: ring buffer, causal notch, sync mode (the UI announces stimulus onsets; required for
the template methods) or async mode (sliding window, FBCCA), dynamic stopping, per-decision timing, FLOP estimate.

## Dataset notes
- D1 (Nakanishi): only Oz is labelled in the dataset documentation; the other channel names are the usual 10-20
  equivalents of the cap positions (approximate).
- D4 (Wearable): the electrode axis of the EEG array is index 0 = dry, index 1 = wet. The dataset readme lists the
  opposite order for the impedance file, but impedances, 50 Hz power and DC offsets all identify index 0 as the dry
  headband (`e5_d4_checks.py`).
