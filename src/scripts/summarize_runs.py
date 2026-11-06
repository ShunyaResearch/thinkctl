"""Collect logs/runs/*/summary.json into one table. Usage: python -I src/scripts/summarize_runs.py [--since YYYYmmdd-HHMMSS]"""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
COLS = ["run", "subject", "script", "combine", "seconds", "attempts", "correct_picks", "wrong_picks", "false_activations",
        "misses", "task_done", "task_success", "decision_latency_ms_median", "decoder_cpu_ms_median", "dropped_pct",
        "max_interval_ms", "stopped", "kill_switch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    ap.add_argument("--out", default=str(ROOT / "logs" / "ckpt2" / "runs_summary.csv"))
    a = ap.parse_args()
    rows = []
    for d in sorted((ROOT / "logs" / "runs").iterdir()):
        if d.name < a.since or not (d / "summary.json").exists():
            continue
        s = json.load(open(d / "summary.json"))
        fr = s.get("frames") if isinstance(s.get("frames"), dict) else {}
        rows.append(dict(run=d.name, subject=s.get("subject"), script=s.get("script", "email"), combine=s.get("combine"),
                         seconds=s["seconds"], attempts=s["attempts"], correct_picks=s["correct_picks"],
                         wrong_picks=s["wrong_picks"], false_activations=s["false_activations"], misses=s["misses"],
                         task_done=s["task_done"], task_success=s["task_success"],
                         decision_latency_ms_median=round(s["decision_latency_ms_median"] or 0, 1),
                         decoder_cpu_ms_median=round(s["decoder_cpu_ms_median"] or 0, 2),
                         dropped_pct=fr.get("dropped_pct"), max_interval_ms=round(fr.get("max_interval_ms", 0), 1),
                         stopped=bool(s.get("events")), kill_switch=s.get("kill_switch")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" | ".join(str(r[c]) for c in COLS))


if __name__ == "__main__":
    main()
