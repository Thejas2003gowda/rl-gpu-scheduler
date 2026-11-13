"""Train and compare run-time predictors; write predicted run times for one virtual cluster.

Splits (same as everywhere else): the virtual cluster's jobs are split by
submit time into training (first 70%) and test (last 30%); the last 15% of
training is the validation slice.

Which model predicts which period (no period ever sees its own labels):
  * test:        model trained on every job (all virtual clusters) whose run
                 time was known when the test period began.
  * validation:  model trained on run times known when validation began.
                 Model choices are made here, on validation only.
  * fit (rest of training): 5 time blocks; each block is predicted by a model
                 trained on run times known at the start of validation,
                 leaving that block out. These are only used as the agent's
                 input during RL training, so that the agent sees predictions
                 with realistic errors instead of in-sample ones.

Decision rule (fixed before looking at results): the "predicted" information
level is kept for RL experiments only if its pair-order accuracy on the
validation slice is at least 3 points higher than the user average's.

Usage:
    python scripts/build_predictions.py --vc 6214e9
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched import read_jobs_csv, time_split  # noqa: E402
from gpusched.predict import (FEATURES, error_factor, fit_model, history_features,  # noqa: E402
                              label_available, pair_order_accuracy, simple_predictions, spearman)

KEEP_RULE_POINTS = 3.0
LABELS = {"user_mean": "User average (current)", "user_geomean": "User geometric mean",
          "user_median": "User median", "user_last2": "User's last 2 jobs",
          "user_gpu_mean": "User average, same GPU count", "model": "Boosted-tree model"}


def score(pred_log, true_log):
    return {"error_factor": error_factor(pred_log, true_log),
            "pair_order_acc": 100 * pair_order_accuracy(pred_log, true_log),
            "spearman": spearman(pred_log, true_log)}


def design(all_jobs, feats, rows, use_vc):
    X = feats.loc[rows, FEATURES].copy()
    if use_vc:
        X["vc_code"] = all_jobs.loc[rows, "vc"].astype("category").cat.codes
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-csv", default="data/philly_jobs.csv")
    ap.add_argument("--vc", default="6214e9")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out", default="results/predictor")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_jobs = read_jobs_csv(args.jobs_csv)
    print("Building features for all jobs ...")
    feats = history_features(all_jobs)
    log_true_all = np.log(all_jobs["duration"].to_numpy())

    vc_idx = all_jobs.index[all_jobs["vc"] == args.vc]
    vc_jobs = all_jobs.loc[vc_idx].sort_values("submit_time", kind="stable")
    train, test = time_split(vc_jobs.reset_index(), args.train_frac)  # keeps original index in "index"
    n_val = int(len(train) * args.val_frac)
    fit, val = train.iloc[:-n_val], train.iloc[-n_val:]
    idx_fit, idx_val, idx_test = fit["index"].to_numpy(), val["index"].to_numpy(), test["index"].to_numpy()
    val_start, test_start = float(val["submit_time"].min()), float(test["submit_time"].min())

    def simple_for(rows, cutoff):
        known = label_available(all_jobs, cutoff)
        fallback = float(np.median(log_true_all[known]))
        preds = simple_predictions(feats.loc[rows], all_jobs.loc[rows, "est_duration"].to_numpy(), fallback)
        return preds

    # ---------------- model selection on validation ----------------
    known_val = np.flatnonzero(label_available(all_jobs, val_start))
    known_val_vc = np.intersect1d(known_val, vc_idx)
    grid = list(itertools.product([True, False], ["absolute_error", "squared_error"], [15, 63]))
    sel_rows = []
    best = None
    for all_vcs, loss, leaves in grid:
        rows = known_val if all_vcs else known_val_vc
        model = fit_model(design(all_jobs, feats, rows, all_vcs), log_true_all[rows], seed=args.seed,
                          loss=loss, max_leaf_nodes=leaves)
        p = model.predict(design(all_jobs, feats, idx_val, all_vcs))
        s = score(p, log_true_all[idx_val])
        sel_rows.append({"train_on_all_vcs": all_vcs, "loss": loss, "max_leaf_nodes": leaves,
                         "train_rows": len(rows), **s})
        if best is None or s["pair_order_acc"] > best[0]:
            best = (s["pair_order_acc"], dict(all_vcs=all_vcs, loss=loss, max_leaf_nodes=leaves), model, p)
        print(f"  all_vcs={all_vcs!s:5} loss={loss:14} leaves={leaves:3} -> "
              f"pair acc {s['pair_order_acc']:.1f}, error x{s['error_factor']:.2f}")
    pd.DataFrame(sel_rows).to_csv(out / "model_selection_validation.csv", index=False)
    cfg = best[1]
    print("Chosen on validation:", cfg)

    def train_model(rows):
        return fit_model(design(all_jobs, feats, rows, cfg["all_vcs"]), log_true_all[rows], seed=args.seed,
                         loss=cfg["loss"], max_leaf_nodes=cfg["max_leaf_nodes"])

    def rows_for(cutoff):
        known = np.flatnonzero(label_available(all_jobs, cutoff))
        return known if cfg["all_vcs"] else np.intersect1d(known, vc_idx)

    # ---------------- predictions per period ----------------
    pred_val = best[3]
    model_test = train_model(rows_for(test_start))
    pred_test = model_test.predict(design(all_jobs, feats, idx_test, cfg["all_vcs"]))
    pred_fit = np.empty(len(idx_fit))
    blocks = np.array_split(np.arange(len(idx_fit)), args.blocks)
    known_v = rows_for(val_start)
    for b in blocks:
        lo, hi = fit["submit_time"].iloc[b[0]], fit["submit_time"].iloc[b[-1]]
        st = all_jobs["submit_time"].to_numpy()[known_v]
        rows = known_v[(st < lo) | (st > hi)]
        pred_fit[b] = train_model(rows).predict(design(all_jobs, feats, idx_fit[b], cfg["all_vcs"]))

    # ---------------- compare all predictors ----------------
    results = []
    for period, rows, cutoff, model_pred in [("validation", idx_val, val_start, pred_val),
                                             ("test", idx_test, test_start, pred_test)]:
        preds = simple_for(rows, cutoff)
        preds["model"] = model_pred
        for name, p in preds.items():
            results.append({"period": period, "predictor": name, **score(p, log_true_all[rows])})
    res = pd.DataFrame(results)
    res.to_csv(out / "comparison.csv", index=False)

    v = res[res.period == "validation"].set_index("predictor")
    gain = v.loc["model", "pair_order_acc"] - v.loc["user_mean", "pair_order_acc"]
    keep = bool(gain >= KEEP_RULE_POINTS)

    # Which features matter (permutation importance on validation)
    from sklearn.inspection import permutation_importance
    Xv = design(all_jobs, feats, idx_val, cfg["all_vcs"])
    pi = permutation_importance(best[2], Xv, log_true_all[idx_val], n_repeats=3, random_state=args.seed,
                                scoring="neg_median_absolute_error")
    imp = pd.Series(pi.importances_mean, index=Xv.columns).sort_values(ascending=False)
    imp.to_csv(out / "feature_importance_validation.csv", header=["drop_in_score_when_shuffled"])

    # ---------------- write the job table with predictions ----------------
    vc_out = all_jobs.loc[np.concatenate([idx_fit, idx_val, idx_test])].copy()
    vc_out["pred_duration"] = np.exp(np.concatenate([pred_fit, pred_val, pred_test]))
    vc_out["pred_source"] = ["crossfit"] * len(idx_fit) + ["validation_model"] * len(idx_val) + \
        ["test_model"] * len(idx_test)
    out_csv = Path(args.out_csv or f"data/philly_jobs_pred_{args.vc}.csv")
    vc_out.to_csv(out_csv, index=False)

    lines = [f"Run-time predictors, virtual cluster {args.vc} "
             f"(validation {len(idx_val):,} jobs, test {len(idx_test):,} jobs)", "",
             "Error factor: median of max(pred/true, true/pred) (1 = perfect). "
             "Pair-order accuracy: share of nearby job pairs (< 10 apart in the queue, run times "
             "differing by >= 10%) put in the right order (50% = coin flip). Higher is better.", "",
             "| Predictor | Val. error factor | Val. pair-order acc. | Test error factor | Test pair-order acc. "
             "| Test Spearman |", "|---|---|---|---|---|---|"]
    t = res[res.period == "test"].set_index("predictor")
    for name in ["user_mean", "user_geomean", "user_median", "user_last2", "user_gpu_mean", "model"]:
        lines.append(f"| {LABELS[name]} | x{v.loc[name, 'error_factor']:.2f} | {v.loc[name, 'pair_order_acc']:.1f}% "
                     f"| x{t.loc[name, 'error_factor']:.2f} | {t.loc[name, 'pair_order_acc']:.1f}% "
                     f"| {t.loc[name, 'spearman']:.2f} |")
    lines += ["", f"Model chosen on validation: {cfg}",
              f"Decision rule (pre-set): keep the 'predicted' level if validation pair-order accuracy "
              f"beats the user average by >= {KEEP_RULE_POINTS:.0f} points. Gain = {gain:+.1f} points "
              f"-> {'KEEP' if keep else 'DROP'}.", "",
              "Top features (validation, permutation importance):",
              ", ".join(f"{k} ({v_:.3f})" for k, v_ in imp.head(6).items())]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    (out / "decision.json").write_text(json.dumps({"gain_points": gain, "keep_predicted_level": keep,
                                                   "model_config": cfg}, indent=2))
    print("\n" + text)
    print(f"\nSaved {out_csv} and {out}/")


if __name__ == "__main__":
    main()
