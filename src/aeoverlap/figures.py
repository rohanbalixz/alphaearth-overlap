"""Run: python -m aeoverlap.figures. Writes results/figures/:

fig1_preregistered.png  (a) how much of each target state the source covers, in the full feature space vs
                        the yield-relevant subspace; (b) subspace coverage vs ranking loss; (c) overlap vs
                        simpler predictors.
fig2_exploratory.png    F1-F3 (not pre-registered): where the loss falls, irrigation in Nebraska, random
                        forest vs ridge, and how much of each feature set is location.

Colors follow the entity (input type), never rank; marker shape or direct labels give a second encoding."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import DATA, EXPLORATORY, FIGURES, PREREG  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {  # fixed categorical slots 1-3 of the reference palette (validated all-pairs for 3 series)
    "AEF": dict(color="#2a78d6", marker="o", label="AlphaEarth"),
    "RS-noclim": dict(color="#eb6834", marker="s", label="Landsat harmonics"),
    "clim": dict(color="#1baf7a", marker="^", label="Climate only"),
}
REF = "#8a8984"   # neutral reference series (yield, comparison model): not a categorical slot


def _style_axes(axes):
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)


def _rc():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})


def fig1() -> None:
    rs_file = DATA / "rs_features.csv"
    if rs_file.exists() and len(pd.read_csv(rs_file, nrows=1).columns) - 2 == 20:
        STYLE["RS-noclim"]["label"] = "Landsat GCVI+NDVI (lite)"          # PREREGISTRATION Amendment 1
    pairs = pd.read_csv(PREREG / "e3_e4_pairs.csv")
    tests = json.loads((PREREG / "e5_e6_tests.json").read_text())
    arms = [a for a in STYLE if a in set(pairs.arm)]
    _rc()
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

    _style_axes((ax1, ax2, ax3))
    fig.text(0.01, 0.005,
             "County corn yield, 12 Corn Belt states, 2017–2024 (data: Ma, Shen, Swatantran & Lobell 2026). "
             "Random forest, 200 trees, 5 seeds. Overlap uses no target yields. "
             "AEF+climate is ~identical to AEF and omitted.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig1_preregistered.png", dpi=200)
    print("wrote results/figures/fig1_preregistered.png")


def fig2() -> None:
    """Exploratory F1-F3. Reads results/exploratory/."""
    s = json.loads((EXPLORATORY / "f1_f3_summary.json").read_text())
    f1 = pd.read_csv(EXPLORATORY / "f1_pairs.csv")
    vs = pd.read_csv(EXPLORATORY / "f3_variance_shares.csv")
    _rc()
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.2))
    (a, b), (c, d) = axs
    blue = STYLE["AEF"]["color"]

    # (a) F1: ranking loss on covered vs uncovered county-years, per pair
    f = f1.dropna(subset=["loss_rank_covered", "loss_rank_uncovered"])
    ne = f.tgt == "NE"
    a.plot([0, 1.6], [0, 1.6], color=INK2, lw=0.8, zorder=2)
    a.scatter(f.loss_rank_uncovered[~ne], f.loss_rank_covered[~ne], s=40, color=blue, alpha=0.7,
              edgecolors=SURF, linewidths=1.2, zorder=3, label="Other targets")
    a.scatter(f.loss_rank_uncovered[ne], f.loss_rank_covered[ne], s=60, color=INK, marker="D",
              edgecolors=SURF, linewidths=1.2, zorder=4, label="Target = Nebraska")
    c1 = s["F1_concept_shift"]
    a.text(0.02, 0.97, f"Pooled: {c1['pooled_share_of_ranking_loss_on_covered']:.0%} of the ranking loss falls on\n"
           f"the {c1['pooled_share_of_county_years_covered']:.0%} of county-years the source covers",
           transform=a.transAxes, va="top", fontsize=8.5, color=INK)
    a.set_xlim(0, 1.6), a.set_ylim(0, 1.6)
    a.set_xlabel("Ranking loss on county-years the source does NOT cover")
    a.set_ylabel("Ranking loss on county-years the source covers")
    a.set_title("(a) Coverage does not protect a county-year (AlphaEarth)", loc="left", fontsize=10.5)
    a.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)

    # (b) F1: irrigation and the transfer-specific error, target = Nebraska, by source state
    ne_s = s["F1_irrigation"]["within_target_irrigation"]["NE_target"]
    src = sorted(ne_s["by_source"], key=lambda k: ne_s["by_source"][k]["relative"])
    yy = np.arange(len(src))
    b.axvline(0, color=INK2, lw=0.8)
    b.scatter([ne_s["by_source"][k]["residual"] for k in src], yy, s=40, color=REF, marker="o", zorder=3,
              label="Transfer residual")
    b.scatter([ne_s["by_source"][k]["relative"] for k in src], yy, s=50, color=blue, marker="D", zorder=4,
              label="Transfer residual minus Nebraska model's residual")
    b.set_yticks(yy, [f"{k} → NE" for k in src], fontsize=8.5)
    b.set_xlim(-0.05, 0.5)
    b.set_xlabel("Spearman ρ with county irrigation share (> 0: irrigated counties under-predicted)")
    b.set_title(f"(b) Irrigation in Nebraska: {ne_s['n_relative_positive']} of {ne_s['n']} sources, "
                f"sign test p = {ne_s['sign_test_p']:.2g}", loc="left", fontsize=10.5)
    b.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK, bbox_to_anchor=(0.12, 0.62))

    # (c) F2: random forest vs ridge, median losses over 132 pairs (AlphaEarth)
    m = s["F2_ridge"]["AEF"]["median"]
    cats = [("Level loss", "loss_level"), ("Ranking loss", "loss_rank")]
    x = np.arange(len(cats))
    for off, model, col in [(-0.19, "rf", blue), (0.19, "ridge", REF)]:
        vals = [m[k][model] for _, k in cats]
        c.bar(x + off, vals, width=0.36, color=col, zorder=3,
              label="Random forest" if model == "rf" else "Ridge regression")
        for xi, v in zip(x + off, vals):
            c.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    e1 = {(r["arm"], r["direction"]): r for r in s["F2_ridge"]["E1"]}
    e1_rf = pd.read_csv(PREREG / "e1_region_transfer.csv").set_index(["arm", "direction"])
    c.text(0.02, 0.97, f"East→West mean bias: ridge {e1[('AEF', 'East->West')]['bias']:+.0f} bu/ac, "
           f"random forest {e1_rf.loc[('AEF', 'East->West'), 'bias']:+.0f} bu/ac",
           transform=c.transAxes, va="top", fontsize=8.5, color=INK)
    c.set_xticks(x, [n for n, _ in cats])
    c.set_ylim(0, 0.75)
    c.set_ylabel("Median over 132 pairs (R² points)")
    c.set_title("(c) The loss is not a random-forest artifact", loc="left", fontsize=10.5)
    c.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)

    # (d) F3: variance shares, median over features in each group
    sh = vs.groupby("group")[["between_state", "between_county", "location_surface"]].median()
    rows = [("Between states", "between_state"), ("Between counties\n(fixed over years)", "between_county"),
            ("Smooth surface in\nlatitude/longitude", "location_surface")]
    groups = [("AEF", "AlphaEarth bands", blue), ("clim", "Climate variables", STYLE["clim"]["color"]),
              ("yield", "Yield", REF)]
    yy = np.arange(len(rows))
    for j, (g, lab, col) in enumerate(groups):
        vals = [sh.loc[g, k] for _, k in rows]
        d.barh(yy + (j - 1) * 0.26, vals, height=0.24, color=col, zorder=3, label=lab)
        for yi, v in zip(yy + (j - 1) * 0.26, vals):
            d.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8, color=INK)
    sk = s["F3_geography"]["skill"]["AEF"]
    d.text(0.985, 0.03, f"AlphaEarth yield skill:\nbetween counties R² {sk['county_mean_R2']:.2f}\n"
           f"within a county, year to year,\nR² {sk['within_county_anomaly_R2']:.2f}",
           transform=d.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
    d.set_yticks(yy, [r[0] for r in rows], fontsize=9)
    d.invert_yaxis()
    d.set_xlim(0, 1.3)
    d.set_xticks(np.linspace(0, 1, 6))
    d.set_xlabel("Share of variance (median over the group's variables)")
    d.set_title("(d) AlphaEarth is mostly a fixed description of place", loc="left", fontsize=10.5)
    d.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)

    _style_axes((a, b, c, d))
    fig.text(0.01, 0.005, "EXPLORATORY (not pre-registered). County corn yield, 12 Corn Belt states, 2017–2024 "
             "(data: Ma, Shen, Swatantran & Lobell 2026; irrigation: USGS 2015).", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig2_exploratory.png", dpi=200)
    print("wrote results/figures/fig2_exploratory.png")


def main() -> None:
    fig1()
    if (EXPLORATORY / "f1_f3_summary.json").exists():
        fig2()


if __name__ == "__main__":
    main()
