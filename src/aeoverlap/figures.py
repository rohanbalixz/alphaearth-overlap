"""Run: python -m aeoverlap.figures. Writes results/figure.png:
(a) how much of each target state the source covers, in the full feature space vs the yield-relevant
subspace; (b) subspace coverage vs the ranking part of transfer loss; (c) overlap vs simpler predictors.
Colors follow the input type (fixed, never by rank); marker shape is a second encoding."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import DATA, RESULTS  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {  # fixed categorical slots 1-3 of the reference palette (validated all-pairs for 3 series)
    "AEF": dict(color="#2a78d6", marker="o", label="AlphaEarth"),
    "RS-noclim": dict(color="#eb6834", marker="s", label="Landsat harmonics"),
    "clim": dict(color="#1baf7a", marker="^", label="Climate only"),
}


def main() -> None:
    rs_file = DATA / "rs_features.csv"
    if rs_file.exists() and len(pd.read_csv(rs_file, nrows=1).columns) - 2 == 20:
        STYLE["RS-noclim"]["label"] = "Landsat GCVI+NDVI (lite)"          # PREREGISTRATION Amendment 1
    pairs = pd.read_csv(RESULTS / "pairs.csv")
    tests = json.loads((RESULTS / "tests.json").read_text())
    arms = [a for a in STYLE if a in set(pairs.arm)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.8), gridspec_kw=dict(width_ratios=[1, 1.25, 0.95]))
    rng = np.random.default_rng(0)

    # (a) coverage of target by source, full space vs yield subspace
    rows = []
    for a in arms:
        rows += [(f"{STYLE[a]['label']}\nall features", a, "full_cover"),
                 (f"{STYLE[a]['label']}\nyield-relevant", a, "sub_cover")]
    for i, (_, a, col) in enumerate(rows):
        v = pairs.loc[pairs.arm == a, col].values
        y = i + rng.uniform(-0.22, 0.22, len(v))
        ax1.scatter(v, y, s=22, color=STYLE[a]["color"], marker=STYLE[a]["marker"], alpha=0.55,
                    linewidths=0.6, edgecolors=SURF, zorder=3)
        ax1.plot([np.median(v)] * 2, [i - 0.32, i + 0.32], color=INK, lw=2, zorder=4, solid_capstyle="round")
    ax1.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax1.invert_yaxis()
    ax1.set_xlim(-0.03, 1.03)
    ax1.set_xlabel("Share of target counties covered by the source state")
    ax1.set_title("(a) Overlap, 132 state pairs (black bar = median)", loc="left", fontsize=10.5)

    # (b) subspace coverage vs ranking loss; extreme points are clipped to a marked ceiling
    CLIP = 1.2
    clipped = []
    for a in arms:
        d = pairs[pairs.arm == a]
        t = tests[a]["P2_sub_cover_vs_loss_rank"]
        inside = d.loss_rank <= CLIP
        ax2.scatter(d.sub_cover[inside], d.loss_rank[inside], s=40, color=STYLE[a]["color"],
                    marker=STYLE[a]["marker"], alpha=0.75, linewidths=1.2, edgecolors=SURF, zorder=3,
                    label=f"{STYLE[a]['label']}  (Spearman ρ = {t['rho']:.2f}, permutation p = {t['p_qap']:.2g})")
        if (~inside).any():
            ax2.scatter(d.sub_cover[~inside], np.full((~inside).sum(), CLIP + 0.04), s=40, marker="^",
                        facecolors="none", edgecolors=STYLE[a]["color"], linewidths=1.2, zorder=3)
            clipped.append(f"{(~inside).sum()} {STYLE[a]['label'].lower()} pairs above {CLIP} "
                           f"(max {d.loss_rank.max():.1f}) shown as open markers")
    ax2.axhline(0, color=INK2, lw=0.8, zorder=2)
    ax2.axhline(CLIP + 0.04, color=GRID, lw=0.8, zorder=1)
    ax2.set_ylim(0, 1.62)
    if clipped:
        ax2.text(0.0, CLIP + 0.09, "; ".join(clipped), fontsize=7.5, color=INK2)
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_xlabel("Overlap in yield-relevant subspace (share of target covered)")
    ax2.set_ylabel("Ranking loss in R² (own-state model − transfer,\nafter matching the target's mean yield)")
    ax2.set_title("(b) Does overlap predict what a transfer loses?", loc="left", fontsize=10.5)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)

    # (c) horse race on the AEF arm: strength of association with ranking loss (all label-free)
    sec = tests["AEF"]["secondary_rho"]
    race = [("Overlap, yield-relevant subspace\n(pre-registered)", "sub_cover"),
            ("Overlap, all features (one-sided)", "full_cover"),
            ("Nearest-neighbour coverage", "knn_cover"),
            ("Distance between states", "geo_km"),
            ("Growing-degree-day difference", "gdd_dist"),
            ("Source sample size", "log_n_src")]
    vals = [abs(sec[f"{k}~loss_rank"]["rho"]) for _, k in race]
    ps = [sec[f"{k}~loss_rank"]["p_qap"] for _, k in race]
    if "P2_sub_cover_vs_loss_rank" in tests["AEF"]:          # use the 9,999-permutation p for the primary
        ps[0] = tests["AEF"]["P2_sub_cover_vs_loss_rank"]["p_qap"]
    yy = np.arange(len(race))
    ax3.barh(yy, vals, height=0.55, color=STYLE["AEF"]["color"], zorder=3)
    for y_, v, p in zip(yy, vals, ps):
        ax3.text(v + 0.012, y_, f"{v:.2f}  (p = {p:.2g})", va="center", fontsize=8.5, color=INK)
    ax3.set_yticks(yy, [r[0] for r in race], fontsize=8.5)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 0.62)
    ax3.set_xlabel("|Spearman ρ| with ranking loss (AlphaEarth)")
    ax3.set_title("(c) Overlap vs simpler predictors", loc="left", fontsize=10.5)

    for ax in (ax1, ax2, ax3):
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005,
             "County corn yield, 12 Corn Belt states, 2017–2024 (data: Ma, Shen, Swatantran & Lobell 2026). "
             "Random forest, 200 trees, 5 seeds. Overlap uses no target yields. "
             "AEF+climate is ~identical to AEF and omitted.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "figure.png", dpi=200)
    print("wrote results/figure.png")


if __name__ == "__main__":
    main()
