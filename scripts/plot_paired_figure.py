"""Draw the proposal's paired-comparison figure from a paired.csv results file.

Usage:
    python scripts/plot_paired_figure.py \
        --paired results/philly_6214e9_c256_warm_nonoverlap/paired.csv \
        --out results/philly_6214e9_c256_warm_nonoverlap/paired_figure.pdf

Each point is a rule's change in total waiting time vs. FIFO first-fit, with
its 95% paired bootstrap CI. Filled marker = CI excludes 0; hollow = CI
includes 0. Sized for one IEEE column (3.5 in).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {"random": "Random", "fifo": "FIFO (strict)", "sgf": "Smallest-GPU-first",
          "sjf": "SJF (user average)", "sjf_oracle": "SJF (true duration)"}
ORDER = ["random", "fifo", "sgf", "sjf", "sjf_oracle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = pd.read_csv(args.paired).set_index("policy").loc[ORDER]
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.45, 1.38))
    ys = list(range(len(p)))[::-1]
    label_x = p["pct_ci_hi"].max() + 2  # one aligned column of values, clear of every line
    for y, (name, r) in zip(ys, p.iterrows()):
        clear = r["pct_ci_hi"] < 0 or r["pct_ci_lo"] > 0
        ax.plot([r["pct_ci_lo"], r["pct_ci_hi"]], [y, y], color="black", lw=1)
        ax.plot(r["pct_change"], y, "o", ms=4.5, color="black", mfc="black" if clear else "white")
        label = f"{r['pct_change']:+.1f}%".replace("-", "\u2212")
        ax.text(label_x, y, label, va="center", fontsize=7)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(ys, [LABELS[n] for n in ORDER])
    ax.set_xlabel("Change in total wait vs. first-fit (%)")
    lo, hi = p["pct_ci_lo"].min(), p["pct_ci_hi"].max()
    ax.set_xlim(lo - 2, label_x + 10)
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
