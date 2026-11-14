"""Summarize the capability checks (batches A_small, A2-A6) over seeds and compare
them with the original single-seed checks in results/validation/batch*.csv.

All numbers are on the 5 validation episodes, with true run times given to the
agents: % change in total wait vs first-fit (negative = less waiting). "Best" is
the best of the checkpoints evaluated during training (optimistic, since it is
picked on the same episodes); "final" is the last checkpoint.

    python scripts/summarize_capability.py --runs runs/capability --out results/validation/rerun
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORIGINAL = {"A": "batchA_flat_network.csv", "A2": "batchA2_slot_network.csv", "A3": "batchA3_dqn_dueling.csv",
            "A4": "batchA4_dqn_wait.csv", "A5": "batchA5_dqn_stable.csv"}


def load_runs(root):
    rows = []
    for f in sorted(Path(root).glob("*/*/seed*/summary.json")):
        run = f.parent
        s = json.loads(f.read_text())
        curve = pd.read_csv(run / "learning_curve.csv")
        cfg = json.loads((run / "config.json").read_text())
        rows.append({"batch": run.parent.parent.name, "setting": run.parent.name, "seed": run.name,
                     "best_val_change_pct": s["best_val_change_pct"], "best_step": s["best_step"],
                     "final_val_change_pct": float(curve["val_change_pct"].iloc[-1]),
                     "wait_share_at_best": s.get("best_val_wait_share", 0.0),
                     "sjf_true_val_change_pct": s["val_sjf_true_change_pct"],
                     "arch": cfg["hyperparameters"].get("arch"), "gamma": cfg["hyperparameters"].get("gamma"),
                     "no_wait": cfg.get("no_wait", False)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default="runs/capability")
    ap.add_argument("--original", default="results/validation")
    ap.add_argument("--out", default="results/validation/rerun")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if runs.empty:
        raise SystemExit(f"no finished runs under {args.runs}")
    orig = {}
    for batch, name in ORIGINAL.items():
        p = Path(args.original) / name
        if p.exists():
            for _, r in pd.read_csv(p).iterrows():
                orig[(batch, r["run"].split("/")[0])] = (r["best val change %"], r["last val change %"])

    def fmt(x):
        return f"{np.mean(x):+.1f} [{np.min(x):+.1f}, {np.max(x):+.1f}]"

    table, lines = [], []
    for (batch, setting), g in runs.groupby(["batch", "setting"], sort=True):
        o = orig.get((batch, setting))
        rec = {"batch": batch, "setting": setting, "seeds": len(g),
               "best_mean": g["best_val_change_pct"].mean(), "best_min": g["best_val_change_pct"].min(),
               "best_max": g["best_val_change_pct"].max(), "final_mean": g["final_val_change_pct"].mean(),
               "final_min": g["final_val_change_pct"].min(), "final_max": g["final_val_change_pct"].max(),
               "wait_share_at_best_mean": g["wait_share_at_best"].mean(),
               "sjf_true_val_change_pct": g["sjf_true_val_change_pct"].iloc[0],
               "original_best": o[0] if o else np.nan, "original_final": o[1] if o else np.nan}
        table.append(rec)
        lines.append(f"| {batch} | {setting} | {len(g)} | {fmt(g['best_val_change_pct'])} "
                     f"| {fmt(g['final_val_change_pct'])} | {g['wait_share_at_best'].mean():.2f} "
                     f"| {rec['sjf_true_val_change_pct']:+.1f} "
                     f"| {f'{o[0]:+.1f} / {o[1]:+.1f}' if o else '-'} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "per_run.csv", index=False)
    pd.DataFrame(table).to_csv(out / "per_setting.csv", index=False)
    md = ["# Capability checks over seeds (validation only, true run times)", "",
          "% change in total wait vs first-fit on the 5 validation episodes (negative = less waiting): "
          "mean [min, max] over seeds. Best = best checkpoint (picked on these same episodes, so "
          "optimistic); final = last checkpoint. WAIT = share of decisions that chose WAIT at the best "
          "checkpoint. Original = the single-seed check that fixed configs/ (best / final).", "",
          "| Batch | Setting | Seeds | Best | Final | WAIT | SJF(true) | Original |",
          "|---|---|---|---|---|---|---|---|", *lines]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
