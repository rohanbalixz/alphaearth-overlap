"""Overlap measures for an ordered pair (source s -> target t).

None of the measures uses target yields, except ``yield_range_cover``, which is an explanatory
check for the East/West asymmetry (P5) and is not a deployable diagnostic.

Primary (pre-registered): ``sub_cover``, the share of target county-years whose cross-fitted,
class-balanced propensity P(source | z) >= 0.05, where z holds the first K = 3 PLS components
fitted on the SOURCE state's features against yield. The measure is one-sided: a target point that
looks like the source is fine; only target points the source never covers are a positivity problem.
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 0.05     # the 0.05/0.95 trimming threshold of Kluger, Di Tommaso & Lobell (2025)
K_PLS = 3


def _crossfit_propensity(X, d, groups, C=1.0):
    """Out-of-fold P(source | x). Folds hold out whole counties; classes are balanced."""
    e = np.zeros(len(d))
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    for tr, te in GroupKFold(5).split(X, d, groups):
        e[te] = clf.fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
    return e


def _summ(e, d):
    et = e[d == 0]
    return dict(auc=roc_auc_score(d, e),
                cover=float(np.mean(et >= EPS)),                       # one-sided
                share2s=float(np.mean((et >= EPS) & (et <= 1 - EPS))))  # two-sided (ATO-style)


def full_space(src, tgt, cols):
    """Domain classifier on all features: AUC, one-sided coverage, two-sided [0.05, 0.95] share."""
    X = np.vstack([src[cols].values, tgt[cols].values])
    d = np.r_[np.ones(len(src)), np.zeros(len(tgt))]
    g = np.r_[src.FIPS.values, tgt.FIPS.values]
    return _summ(_crossfit_propensity(X, d, g), d)


def yield_subspace(src, tgt, cols, k=K_PLS):
    """Domain classifier on the source-fitted PLS(k) subspace of yield-relevant directions."""
    sc = StandardScaler().fit(src[cols].values)
    pls = PLSRegression(n_components=k).fit(sc.transform(src[cols].values), src.y.values)
    Z = np.vstack([pls.transform(sc.transform(src[cols].values)),
                   pls.transform(sc.transform(tgt[cols].values))])
    d = np.r_[np.ones(len(src)), np.zeros(len(tgt))]
    g = np.r_[src.FIPS.values, tgt.FIPS.values]
    return _summ(_crossfit_propensity(Z, d, g), d)


def knn_cover(src, tgt, cols, q=0.95):
    """Share of target points whose nearest source point is no farther than the q-quantile of the
    source's own nearest-neighbour distances (neighbour taken from a different county)."""
    sc = StandardScaler().fit(src[cols].values)
    Xs, Xt = sc.transform(src[cols].values), sc.transform(tgt[cols].values)
    dist, idx = NearestNeighbors(n_neighbors=12).fit(Xs).kneighbors(Xs)
    f = src.FIPS.values
    own = np.array([next((dd for dd, j in zip(dist[i], idx[i]) if f[j] != f[i]), dist[i, -1])
                    for i in range(len(Xs))])
    dt, _ = NearestNeighbors(n_neighbors=1).fit(Xs).kneighbors(Xt)
    return float(np.mean(dt[:, 0] <= np.quantile(own, q)))


def yield_range_cover(src, tgt):
    """Share of target county-years whose yield lies inside the source's 5th-95th percentile range."""
    lo, hi = np.percentile(src.y, [5, 95])
    return float(np.mean((tgt.y >= lo) & (tgt.y <= hi)))


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def rivals(src, tgt):
    """Label-free pair descriptors used as rival predictors (P3)."""
    gdd = [f"GDD_{m}" for m in range(5, 10)]
    return dict(
        # cf. Wang, Azzari & Lobell (2019): RF transfer between states degrades as GDD differences grow
        gdd_dist=abs(src[gdd].sum(axis=1).mean() - tgt[gdd].sum(axis=1).mean()),
        geo_km=haversine_km(src.lon.mean(), src.lat.mean(), tgt.lon.mean(), tgt.lat.mean()),
        log_n_src=float(np.log(len(src))),
    )
