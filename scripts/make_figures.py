"""Summary figures, drawn only from committed result files.

    results/rl/B_test/report.json             test period (pre-registered runs)
    results/validation/*.csv                  validation-only capability checks

Writes PNG and PDF versions to results/figures/.

Usage:
    python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

# Colors: validated categorical slots 1-3 (all-pairs safe), text tokens, neutral reference gray.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
REF = "#9b9a95"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "grid.linestyle": "-", "axes.axisbelow": True,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    # no creation date in the PDF, so re-running the script does not change committed files
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def dot(ax, x, y, color, size=8, **kw):
    ax.plot(x, y, "o", color=color, markersize=size, markeredgecolor=SURFACE, markeredgewidth=2, **kw)


def fig_main(r):
    """Main pre-registered test: every seed, IQM with 95% CI, reference rules."""
    base = r["dqn_c256_estimate"]["baseline_change_vs_first_fit_pct"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    rows = [("DQN (own)", "dqn_c256_estimate", BLUE), ("PPO", "ppo_c256_estimate", ORANGE)]
    for i, (label, key, color) in enumerate(rows):
        s = r[key]["success_test_vs_first_fit"]
        seeds = np.array(s["per_seed_change_pct"])
        jitter = np.linspace(-0.12, 0.12, len(seeds))
        ax.plot(seeds, i + jitter, "o", color=color, alpha=0.45, markersize=7, markeredgewidth=0)
        ax.plot([s["ci_lo"], s["ci_hi"]], [i + 0.32] * 2, color=color, solid_capstyle="round")
        dot(ax, s["iqm_change_pct"], i + 0.32, color, size=10)
        verdict = "meets criterion" if s["success"] else "does not meet criterion"
        ax.text(max(s["ci_hi"], seeds.max()) + 0.4, i + 0.1,
                f"IQM {s['iqm_change_pct']:+.1f}%  [{s['ci_lo']:+.1f}, {s['ci_hi']:+.1f}]\n{verdict}",
                va="center", fontsize=10, color=INK2)
    refs = [(0.0, "first-fit", "left"), (base["sjf_pred"], "best realistic rule", "right"),
            (base["sjf_oracle"], "SJF, true run times", "left")]
    for x, label, ha in refs:
        ax.axvline(x, color=REF, linewidth=1.2, linestyle=(0, (4, 3)) if x else "-", zorder=0)
        ax.text(x + (0.15 if ha == "left" else -0.15), 1.8, label, ha=ha, va="center", fontsize=9, color=MUTED)
    ax.set_yticks([0, 1], [r_[0] for r_ in rows])
    ax.set_ylim(-0.45, 2.1)
    ax.set_xlim(base["sjf_oracle"] - 1.5, 13)
    ax.set_xlabel("Change in total waiting time vs first-fit (%)   ← less waiting")
    ax.set_title("Main test: 256 GPUs, user-average run times, 10 seeds (test period)")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    save(fig, "main_test_256")


def fig_information(r):
    """E1: how much each level of run-time information is worth, for SJF and for RL."""
    levels = [("none", "No run\ntimes"), ("estimate", "User\naverage"),
              ("predicted", "Predicted"), ("oracle", "True run\ntimes")]
    rule_of = {"none": None, "estimate": "sjf", "predicted": "sjf_pred", "oracle": "sjf_oracle"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    xs = np.arange(len(levels))
    for ax, cap in zip(axes, (256, 128)):
        base = r[f"ppo_c{cap}_estimate"]["baseline_change_vs_first_fit_pct"]
        rule = [0.0 if rule_of[k] is None else base[rule_of[k]] for k, _ in levels]
        ax.plot(xs - 0.12, rule, "-", color=BLUE, alpha=0.6, zorder=1)
        for x, y in zip(xs - 0.12, rule):
            dot(ax, x, y, BLUE)
        for j, (k, _) in enumerate(levels):
            s = r[f"ppo_c{cap}_{k}"]["success_test_vs_first_fit"]
            ax.plot([j + 0.12] * 2, [s["ci_lo"], s["ci_hi"]], color=ORANGE, solid_capstyle="round")
            dot(ax, j + 0.12, s["iqm_change_pct"], ORANGE)
        d = r[f"dqn_c{cap}_estimate"]["success_test_vs_first_fit"]
        ax.plot([1 + 0.32] * 2, [d["ci_lo"], d["ci_hi"]], color=AQUA, solid_capstyle="round")
        dot(ax, 1 + 0.32, d["iqm_change_pct"], AQUA)
        ax.axhline(0, color=REF, linewidth=1.2, zorder=0)
        ax.set_xticks(xs, [lab for _, lab in levels])
        ax.set_title(f"{cap} GPUs")
        ax.set_xlim(-0.5, len(levels) - 0.4)
    axes[0].set_ylabel("Change in total wait vs first-fit (%)")
    handles = [plt.Line2D([], [], marker="o", color=c, markersize=8, markeredgecolor=SURFACE,
                          markeredgewidth=2, linestyle="") for c in (BLUE, ORANGE, AQUA)]
    fig.legend(handles, ["SJF rule using that information (no information = first-fit)",
                         "PPO, IQM of 5 seeds with 95% CI", "DQN (user average only), IQM with 95% CI"],
               loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.suptitle("What run-time information is worth (test period, lower is better)",
                 x=0.06, y=1.03, ha="left", fontweight="bold", fontsize=13)
    save(fig, "information_levels")


def fig_scarcity(r):
    """RL vs rules as the cluster gets scarcer."""
    caps = [128, 192, 256]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(caps))
    best = []
    for cap in caps:
        b = r[f"dqn_c{cap}_estimate"]["baseline_change_vs_first_fit_pct"]
        best.append(min(b[k] for k in ("sgf", "sjf", "sjf_pred", "random")))
    oracle = [r[f"dqn_c{cap}_estimate"]["baseline_change_vs_first_fit_pct"]["sjf_oracle"] for cap in caps]
    ax.plot(xs, oracle, color=REF, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(xs[-1] + 0.08, oracle[-1], "SJF, true run times", va="center", fontsize=9, color=MUTED)
    ax.plot(xs - 0.14, best, "-", color=BLUE, alpha=0.6)
    for x, y in zip(xs - 0.14, best):
        dot(ax, x, y, BLUE)
    for off, algo, color in [(0.0, "dqn", AQUA), (0.14, "ppo", ORANGE)]:
        for x, cap in zip(xs + off, caps):
            s = r[f"{algo}_c{cap}_estimate"]["success_test_vs_first_fit"]
            ax.plot([x, x], [s["ci_lo"], s["ci_hi"]], color=color, solid_capstyle="round")
            dot(ax, x, s["iqm_change_pct"], color)
    ax.axhline(0, color=REF, linewidth=1.2, zorder=0)
    ax.set_xticks(xs, [f"{c} GPUs" for c in caps])
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("Change in total wait vs first-fit (%)")
    ax.set_title("Scarcer GPUs: where ordering pays (user-average run times)")
    handles = [plt.Line2D([], [], marker="o", color=c, markersize=8, markeredgecolor=SURFACE,
                          markeredgewidth=2, linestyle="") for c in (BLUE, AQUA, ORANGE)]
    ax.legend(handles, ["Best realistic rule", "DQN, IQM with 95% CI", "PPO, IQM with 95% CI"],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)
    save(fig, "scarcity")


def fig_capability():
    """Validation-only capability check: can each network learn SJF with true run times (256 GPUs)?"""
    v = ROOT / "results" / "validation"
    a = pd.read_csv(v / "batchA_flat_network.csv")
    a2 = pd.read_csv(v / "batchA2_slot_network.csv")
    a5 = pd.read_csv(v / "batchA5_dqn_stable.csv")
    flat = a[a.GPUs == 256]
    rows = [
        ("Flat network, PPO (best of 2)", flat[flat.algo == "ppo"]["best val change %"].min()),
        ("Flat network, DQN (best of 2)", flat[flat.algo == "dqn"]["best val change %"].min()),
        ("Per-job scorer, PPO, γ = 1", a2[(a2.GPUs == 256) & (a2.algo == "ppo") & (a2.gamma == 1.0)]["best val change %"].iloc[0]),
        ("Per-job scorer, DQN, no WAIT", a5["best val change %"].min()),
    ]
    sjf = a2["SJF(true) val %"][a2.GPUs == 256].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ys = np.arange(len(rows))
    vals = [v_ for _, v_ in rows]
    ax.barh(ys, vals, height=0.42, color=BLUE, zorder=2)
    for y, v_ in zip(ys, vals):
        if v_ < -5:  # long bar: label inside, at the bar's end
            ax.text(v_ + 0.4, y, f"{v_:+.1f}%", va="center", ha="left", fontsize=10, color=SURFACE, zorder=3)
        else:
            ax.text(v_ - 0.4, y, f"{v_:+.1f}%", va="center", ha="right", fontsize=10, color=INK2)
    ax.axvline(sjf, color=REF, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(sjf + 0.25, -0.75, f"SJF, true run times ({sjf:+.1f}%)", ha="left", fontsize=9, color=MUTED)
    ax.set_yticks(ys, [lab for lab, _ in rows])
    ax.set_ylim(len(rows) - 0.5, -1.0)
    ax.set_xlim(sjf - 5, 1)
    ax.set_xlabel("Best change in total wait vs first-fit, validation (%)   ← less waiting")
    ax.set_title("Can the agent learn SJF when it sees true run times? (256 GPUs)")
    ax.grid(axis="y", visible=False)
    save(fig, "capability_check")


def main():
    r = json.loads((ROOT / "results" / "rl" / "B_test" / "report.json").read_text())
    fig_main(r)
    fig_information(r)
    fig_scarcity(r)
    fig_capability()


if __name__ == "__main__":
    main()
