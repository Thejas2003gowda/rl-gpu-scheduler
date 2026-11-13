"""One-line summary per training run (validation only; never touches the test period).

Usage:
    python scripts/summarize_runs.py runs/
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="runs")
    ap.add_argument("--out", default=None, help="also save the table as CSV")
    args = ap.parse_args()
    rows = []
    for summ in sorted(Path(args.root).glob("**/summary.json")):
        cfg = json.loads((summ.parent / "config.json").read_text())
        s = json.loads(summ.read_text())
        curve = pd.read_csv(summ.parent / "learning_curve.csv")
        rows.append({"run": str(summ.parent.relative_to(args.root)), "algo": cfg["algo"],
                     "GPUs": cfg["capacity"], "info": cfg["duration_info"],
                     "network": cfg["hyperparameters"].get("arch", "mlp"),
                     "gamma": cfg["hyperparameters"].get("gamma", "default"),
                     "best val change %": round(s["best_val_change_pct"], 2),
                     "last val change %": round(float(curve["val_change_pct"].iloc[-1]), 2),
                     "SJF(true) val %": round(s.get("val_sjf_true_change_pct", float("nan")), 2),
                     "WAIT share (best)": round(s.get("best_val_wait_share", float("nan")), 3),
                     "best step": s["best_step"], "minutes": round(s["minutes"], 1)})
    if not rows:
        raise SystemExit(f"no finished runs under {args.root}")
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
