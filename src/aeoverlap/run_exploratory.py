"""Exploratory follow-up analyses. NOT pre-registered: they were designed after the pre-registered results
were known, in response to them (PREREGISTRATION.md: "Any extra analysis will be labelled exploratory").

Run: python -m aeoverlap.run_exploratory      (about 10-15 minutes; writes results/exploratory/)

F1  Concept shift and irrigation. Split each AEF transfer's ranking loss between the target county-years
    the source covers (yield-subspace propensity >= 0.05) and those it does not. Loss on covered
    county-years is failure that overlap cannot see. Test whether irrigation (USGS 2015) explains it.
F2  Model dependence. The 132-pair design with ridge regression, which extrapolates linearly, instead of a
    random forest, which cannot predict outside the range of its training yields.
F3  Geography. How much of each feature set is fixed by location, and how much of AEF's yield skill is
    year-to-year signal within a county.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .data import AEF, CLIM, EAST, EXPLORATORY, PREREG, SEEDS, STATES, WEST, irrigation_share, load, state_frames
from .inference import dsp_partial, qap_spearman
from .overlap import EPS, subspace_propensity
from .run_preregistered import environment_report
from .transfer import metrics, rf

OUT = EXPLORATORY
N_PERM = 9999


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15)))


def _oof(T, cols, make_model):
    """Out-of-fold predictions inside one state, 5 folds holding out whole counties."""
    p = np.zeros(len(T))
    for tr, te in GroupKFold(5).split(T, groups=T.FIPS):
        p[te] = make_model().fit(T.iloc[tr][cols], T.iloc[tr].y).predict(T.iloc[te][cols])
    return p


# ----------------------------------------------------------------------------------------------- F1
def rf_point_predictions(frames, cols):
    """Seed-averaged RF predictions: target-state oracle (out-of-fold) and every transfer (s -> t)."""
    oracle = {t: np.mean([_oof(frames[t], cols, lambda s=seed: rf(s)) for seed in SEEDS], axis=0)
              for t in STATES}
    trans = {}
    for seed in SEEDS:
        for s in STATES:
            m = rf(seed).fit(frames[s][cols], frames[s].y)
            for t in STATES:
                if t != s:
                    trans[(s, t)] = trans.get((s, t), 0) + m.predict(frames[t][cols]) / len(SEEDS)
    return oracle, trans


def f1_concept_shift(frames, cols):
    """Per-county-year excess squared error of the level-matched transfer over the oracle, in units of
    the target's yield variance (its mean over a pair equals that pair's ranking loss), split by coverage."""
    oracle, trans = rf_point_predictions(frames, cols)
    rows, pts = [], []
    for (s, t), p in trans.items():
        S, T = frames[s], frames[t]
        y = T.y.values
        e, d = subspace_propensity(S, T, cols)
        cov = e[d == 0] >= EPS
        p_lvl = p - p.mean() + y.mean()
        exc = ((y - p_lvl) ** 2 - (y - oracle[t]) ** 2) / y.var()
        rows.append(dict(src=s, tgt=t, cover=float(cov.mean()), loss_rank=float(exc.mean()),
                         loss_rank_covered=float(exc[cov].mean()) if cov.any() else np.nan,
                         loss_rank_uncovered=float(exc[~cov].mean()) if (~cov).any() else np.nan,
                         loss_on_covered=float(exc[cov].sum() / exc.sum()) if exc.sum() > 0 else np.nan))
        pts.append(pd.DataFrame(dict(src=s, tgt=t, FIPS=T.FIPS.values, year=T.year.values, covered=cov,
                                     excess=exc, resid=y - p_lvl, resid_oracle=y - oracle[t],
                                     irr_share=T.irr_share.values)))
    return pd.DataFrame(rows), pd.concat(pts, ignore_index=True)


def f1_irrigation(pairs_aef, f1p, pts, irr_state):
    """Does irrigation explain what overlap cannot? Pair level (state means) and county level."""
    a = pairs_aef.copy()
    a["irr_src"], a["irr_tgt"] = a.src.map(irr_state), a.tgt.map(irr_state)
    a["irr_absdiff"] = (a.irr_tgt - a.irr_src).abs()
    a = a.merge(f1p[["src", "tgt", "loss_rank_covered"]], on=["src", "tgt"])
    out = {}
    for x in ["irr_absdiff", "sub_cover", "geo_km"]:
        rho, p = qap_spearman(a, x, "loss_rank", N_PERM)
        out[f"rho_{x}_vs_loss_rank"] = dict(rho=rho, p_qap=p)
    cov = a.dropna(subset=["loss_rank_covered"])
    rho, p = qap_spearman(cov, "irr_absdiff", "loss_rank_covered", N_PERM)
    out["rho_irr_absdiff_vs_loss_rank_covered"] = dict(rho=rho, p_qap=p, n_pairs=len(cov))
    coef, t, p = dsp_partial(a, "irr_absdiff", "loss_rank", ["geo_km", "gdd_dist"], fe=True, n_perm=N_PERM)
    out["partial_irr_absdiff_given_distance_gdd_fe"] = dict(std_coef=coef, t=t, p_dsp_qap=p)
    coef, t, p = dsp_partial(a, "irr_absdiff", "loss_rank", ["geo_km", "gdd_dist", "sub_cover"], fe=True,
                             n_perm=N_PERM)
    out["partial_irr_absdiff_given_distance_gdd_overlap_fe"] = dict(std_coef=coef, t=t, p_dsp_qap=p)

    # County level: within each transfer, are irrigated counties under-predicted? Any model shrinks toward
    # the mean and so under-predicts high-yield (often irrigated) counties; the control is the target's own
    # model. rho_irr_vs_relative uses (transfer residual - oracle residual) = oracle prediction - transfer
    # prediction, i.e. the part of the error that is specific to the transfer.
    within = []
    for (s, t), g in pts.groupby(["src", "tgt"]):
        if g.irr_share.std() > 0:
            within.append(dict(src=s, tgt=t,
                               rho_irr_vs_residual=float(spearmanr(g.irr_share, g.resid).correlation),
                               rho_irr_vs_relative=float(spearmanr(g.irr_share, g.resid - g.resid_oracle).correlation),
                               rho_irr_vs_oracle_residual=float(spearmanr(g.irr_share, g.resid_oracle).correlation)))
    w = pd.DataFrame(within)

    def _summ(sub):
        k = int((sub.rho_irr_vs_relative > 0).sum())
        return dict(n=len(sub), median_rho_irr_vs_residual=float(sub.rho_irr_vs_residual.median()),
                    median_rho_irr_vs_relative=float(sub.rho_irr_vs_relative.median()),
                    rho_irr_vs_oracle_residual=float(sub.rho_irr_vs_oracle_residual.median()),
                    n_relative_positive=k,
                    sign_test_p=float(binomtest(k, len(sub), 0.5, alternative="greater").pvalue),
                    by_source={r.src: dict(residual=round(r.rho_irr_vs_residual, 3),
                                           relative=round(r.rho_irr_vs_relative, 3)) for r in sub.itertuples()})
    out["within_target_irrigation"] = dict(
        all_pairs_median_rho_irr_vs_relative=float(w.rho_irr_vs_relative.median()),
        NE_target=_summ(w[w.tgt == "NE"]), KS_target=_summ(w[w.tgt == "KS"]))
    return out, a, w


# ----------------------------------------------------------------------------------------------- F2
def f2_ridge(df, frames, arms, pairs_rf):
    e1, rows = [], []
    E, W = df[df.st.isin(EAST)], df[df.st.isin(WEST)]
    for arm, cols in arms.items():
        for name, (s, t) in {"East->West": (E, W), "West->East": (W, E)}.items():
            e1.append(dict(arm=arm, direction=name, **metrics(t.y, ridge().fit(s[cols], s.y).predict(t[cols]))))
        orc = {t: r2_score(frames[t].y, _oof(frames[t], cols, ridge)) for t in STATES}
        for s in STATES:
            S = frames[s]
            m = ridge().fit(S[cols], S.y)
            for t in STATES:
                if t != s:
                    p = m.predict(frames[t][cols])
                    rows.append(dict(arm=arm, src=s, tgt=t, R2_oracle=orc[t], **metrics(frames[t].y, p),
                                     below_src_min=float(np.mean(p < S.y.min()))))
    r = pd.DataFrame(rows)
    r["loss_total"], r["loss_level"] = r.R2_oracle - r.R2, r.R2_level - r.R2
    r["loss_rank"] = r.R2_oracle - r.R2_level
    r = r.merge(pairs_rf[["arm", "src", "tgt", "sub_cover", "geo_km"]], on=["arm", "src", "tgt"])
    summ = {}
    for arm in arms:
        rr, rf_ = r[r.arm == arm], pairs_rf[pairs_rf.arm == arm]
        rho, p = qap_spearman(rr, "sub_cover", "loss_rank", N_PERM)
        rho_g, p_g = qap_spearman(rr, "geo_km", "loss_rank", N_PERM)
        summ[arm] = {
            "median": {m: dict(ridge=float(rr[m].median()), rf=float(rf_[m].median()))
                       for m in ["R2", "R2_level", "R2_oracle", "loss_level", "loss_rank"]},
            "share_R2_negative": dict(ridge=float((rr.R2 < 0).mean()), rf=float((rf_.R2 < 0).mean())),
            "median_share_predictions_below_source_min": float(rr.below_src_min.median()),
            "rho_sub_cover_vs_loss_rank": dict(rho=rho, p_qap=p),
            "rho_geo_km_vs_loss_rank": dict(rho=rho_g, p_qap=p_g),
        }
    return pd.DataFrame(e1), r, summ


# ----------------------------------------------------------------------------------------------- F3
def variance_shares(df, cols):
    """Share of each column's variance that lies between states, between counties (time-invariant), and
    on a smooth cubic surface in (lon, lat)."""
    P = PolynomialFeatures(3, include_bias=False).fit_transform(StandardScaler().fit_transform(df[["lon", "lat"]]))
    rows = []
    for c in cols:
        v = df[c].values.astype(float)
        tot = ((v - v.mean()) ** 2).sum()
        st = df.groupby("st")[c].transform("mean").values
        cty = df.groupby("FIPS")[c].transform("mean").values
        surf = LinearRegression().fit(P, v).predict(P)
        rows.append(dict(feature=c, between_state=((st - v.mean()) ** 2).sum() / tot,
                         between_county=((cty - v.mean()) ** 2).sum() / tot,
                         location_surface=1 - ((v - surf) ** 2).sum() / tot))
    return pd.DataFrame(rows)


def anomaly_skill(df, cols):
    """R2 for within-county yield anomalies, leave-one-year-out; county means from training years only."""
    r2s = []
    for seed in SEEDS:
        yt, yp = [], []
        for yr in sorted(df.year.unique()):
            tr, te = df[df.year != yr], df[df.year == yr]
            mx, my = tr.groupby("FIPS")[cols].mean(), tr.groupby("FIPS").y.mean()
            te = te[te.FIPS.isin(mx.index)]
            m = rf(seed).fit(tr[cols].values - mx.loc[tr.FIPS].values, tr.y.values - my.loc[tr.FIPS].values)
            yt.append(te.y.values - my.loc[te.FIPS].values)
            yp.append(m.predict(te[cols].values - mx.loc[te.FIPS].values))
        r2s.append(r2_score(np.concatenate(yt), np.concatenate(yp)))
    return float(np.mean(r2s))


def county_mean_skill(df, cols):
    """R2 for county-mean yield from county-mean features, 5-fold CV over counties (pooled states)."""
    c = df.groupby("FIPS")[cols + ["y"]].mean().reset_index()
    r2s = []
    for seed in SEEDS:
        p = np.zeros(len(c))
        for tr, te in GroupKFold(5).split(c, groups=c.FIPS):
            p[te] = rf(seed).fit(c.iloc[tr][cols], c.iloc[tr].y).predict(c.iloc[te][cols])
        r2s.append(r2_score(c.y, p))
    return float(np.mean(r2s))


def f3_geography(df):
    vs = pd.concat([variance_shares(df, AEF).assign(group="AEF"), variance_shares(df, CLIM).assign(group="clim"),
                    variance_shares(df, ["y"]).assign(group="yield")], ignore_index=True)
    shares = vs.groupby("group")[["between_state", "between_county", "location_surface"]].median()
    y_within = 1 - shares.loc["yield", "between_county"]
    skill = {arm: dict(within_county_anomaly_R2=anomaly_skill(df, cols), county_mean_R2=county_mean_skill(df, cols))
             for arm, cols in {"AEF": AEF, "clim": CLIM, "AEF+clim": AEF + CLIM}.items()}
    return vs, dict(median_variance_shares=shares.round(4).to_dict(orient="index"),
                    yield_within_county_variance_share=float(y_within), skill=skill)


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    import time
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df, arms = load(use_rs=False)
    df["irr_share"] = (df.FIPS.map(irrigation_share()) / df.area_km2).clip(0, 1)
    assert df.irr_share.notna().all(), "irrigation data missing for some counties"
    frames = state_frames(df)
    pairs_rf = pd.read_csv(PREREG / "e3_e4_pairs.csv")
    irr_state = {s: float(frames[s].irr_share.mean()) for s in STATES}
    res = {"_note": "EXPLORATORY: not pre-registered. Designed after the pre-registered results were known.",
           "irrigation_share_state_mean": irr_state}

    f1p, pts = f1_concept_shift(frames, AEF)
    aef = pairs_rf[pairs_rf.arm == "AEF"]
    chk = f1p.merge(aef[["src", "tgt", "sub_cover", "loss_rank"]], on=["src", "tgt"], suffixes=("", "_prereg"))
    assert np.allclose(chk.cover, chk.sub_cover), "coverage must match the pre-registered sub_cover exactly"
    full = f1p[f1p.cover >= 0.9]
    res["F1_concept_shift"] = dict(
        loss_rank_seed_avg_vs_prereg_spearman=float(spearmanr(chk.loss_rank, chk.loss_rank_prereg).correlation),
        median_loss_rank_covered=float(f1p.loss_rank_covered.median()),
        median_loss_rank_uncovered=float(f1p.loss_rank_uncovered.median()),
        pooled_share_of_ranking_loss_on_covered=float(pts.excess[pts.covered].sum() / pts.excess.sum()),
        pooled_share_of_county_years_covered=float(pts.covered.mean()),
        pairs_with_cover_ge_0_9=dict(n=len(full), median_loss_rank=float(full.loss_rank.median()),
                                     worst=full.nlargest(5, "loss_rank")[["src", "tgt", "cover", "loss_rank"]]
                                     .round(3).to_dict(orient="records")))
    irr_out, pair_irr, within = f1_irrigation(aef, f1p, pts, irr_state)
    res["F1_irrigation"] = irr_out
    f1p.to_csv(OUT / "f1_pairs.csv", index=False)
    pts.to_csv(OUT / "f1_county_years.csv.gz", index=False)
    within.to_csv(OUT / "f1_irrigation_within_pairs.csv", index=False)
    print("[F1] done", flush=True)

    e1r, ridge_pairs, f2 = f2_ridge(df, frames, arms, pairs_rf)
    res["F2_ridge"] = dict(E1=e1r[["arm", "direction", "R2", "R2_level", "bias"]].round(3).to_dict(orient="records"),
                           **f2)
    ridge_pairs.to_csv(OUT / "f2_ridge_pairs.csv", index=False)
    e1r.to_csv(OUT / "f2_ridge_e1_region_transfer.csv", index=False)
    print("[F2] done", flush=True)

    vs, f3 = f3_geography(df)
    res["F3_geography"] = f3
    vs.to_csv(OUT / "f3_variance_shares.csv", index=False)
    print("[F3] done", flush=True)

    (OUT / "f1_f3_summary.json").write_text(json.dumps(res, indent=2, default=float))
    (OUT / "environment.txt").write_text(environment_report(t0, time.time()))
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
