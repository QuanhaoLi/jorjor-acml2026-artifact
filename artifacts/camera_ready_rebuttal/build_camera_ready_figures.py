from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "five_run_and_exact_posterior_data.json").read_text())
RUNS = ["k", "l", "m", "n", "o"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _forest(ax, payload, color: str) -> None:
    y_runs = list(range(len(RUNS), 0, -1))
    for y, run in zip(y_runs, RUNS, strict=True):
        estimate, lower, upper = payload["runs"][run]
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=3.8,
            elinewidth=1.0,
            capsize=2.0,
        )
    mean, lower, upper = payload["across_run"]
    ax.errorbar(
        mean,
        0,
        xerr=[[mean - lower], [upper - mean]],
        fmt="D",
        color="black",
        ecolor="black",
        markersize=4.2,
        elinewidth=1.2,
        capsize=2.2,
    )
    ax.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_runs + [0], [f"run {run}" for run in RUNS] + ["run mean"])
    ax.grid(axis="x", color="#d9dee5", linewidth=0.6)
    ax.set_title(payload["label"] + "\n" + payload["summary"])
    ax.set_xlabel("accuracy delta")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_five_run() -> None:
    gates = DATA["five_run_gates"]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.05))
    # G7 is the frozen storage key; the payload label presents the claim-safe S7 name.
    colors = {"G5": "#2f6fab", "G6": "#d17a22", "G7": "#7a4ea3"}
    for ax, gate in zip(axes, ("G5", "G6", "G7"), strict=True):
        _forest(ax, gates[gate], colors[gate])
    fig.suptitle(
        "Five-run robustness of realized-hand evidence",
        fontsize=10.5,
        fontweight="bold",
        y=0.985,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666", label="run estimate + within-run 95% bootstrap CI"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="black", markeredgecolor="black", label="equal-run mean + 95% Student-t CI"),
    ]
    fig.subplots_adjust(left=0.09, right=0.99, top=0.75, bottom=0.25, wspace=0.42)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.savefig(
        ROOT / "figure_five_run_realized_hand_gates.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(ROOT / "figure_five_run_realized_hand_gates.png", dpi=300)
    plt.close(fig)


def build_exact_posterior() -> None:
    exact = DATA["exact_posterior"]
    pointwise = exact["pointwise_kl_improvement"]
    slope = exact["matched_pair_slope"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7), constrained_layout=True)

    _forest(axes[0], pointwise, "#2f6fab")
    axes[0].set_title("A. Pointwise posterior prediction\nKL/CE improvement beyond A/V+composition")
    axes[0].set_xlabel("KL/CE improvement (nats; positive favors hidden)")

    y_runs = list(range(len(RUNS), 0, -1))
    for y, run in zip(y_runs, RUNS, strict=True):
        axes[1].plot(slope["runs"][run], y, "o", color="#7a4ea3", markersize=4.0)
    mean, lower, upper = slope["across_run"]
    axes[1].errorbar(
        mean,
        0,
        xerr=[[mean - lower], [upper - mean]],
        fmt="D",
        color="black",
        ecolor="black",
        markersize=4.2,
        elinewidth=1.2,
        capsize=2.2,
    )
    axes[1].axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_xlim(-0.0005, 0.0105)
    axes[1].set_yticks(y_runs + [0], [f"run {run}" for run in RUNS] + ["run mean"])
    axes[1].grid(axis="x", color="#d9dee5", linewidth=0.6)
    axes[1].set_title("B. Matched posterior geometry\nPair-JS slope is severely compressed")
    axes[1].set_xlabel("Pair-JS slope (faithful reference = 1, off-scale)")
    axes[1].text(
        0.98,
        0.04,
        "Pearson mean = 0.129\nfaithful slope = 1 (off-scale)",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#444444",
    )
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Exact teacher-induced posterior: pointwise gain with compressed pairwise geometry",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.savefig(
        ROOT / "figure_exact_teacher_posterior_evidence.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(ROOT / "figure_exact_teacher_posterior_evidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_five_run()
    build_exact_posterior()
