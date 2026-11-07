"""Transfer metrics.

For target t and predictions p:
  R2        = 1 - SSE / SST about the target mean   (paper Eq. 9)
  R2_level  = R2 after shifting p so its mean equals the target's mean yield. The target mean is
              the kind of aggregate NASS publishes, so this is the part of the loss an aggregate-
              statistics correction (as in Kluger, Wang & Lobell 2021) can remove.
Oracle R2(t): RF trained inside t, 5-fold CV holding out whole counties (unseen places, seen years,
              which matches what a transferred model faces).
Loss decomposition:
  loss_total = R2_oracle - R2
  loss_level = R2_level  - R2          (fixable with the target's mean)
  loss_rank  = R2_oracle - R2_level    (not fixable that way; primary outcome for P2)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from common import N_TREES, SEEDS, STATES


def rf(seed):
    return RandomForestRegressor(n_estimators=N_TREES, random_state=seed, n_jobs=-1)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    p_lvl = p - p.mean() + y.mean()
    return dict(R2=r2_score(y, p), R2_level=r2_score(y, p_lvl),
                r2_corr=float(np.corrcoef(y, p)[0, 1] ** 2),
                rmse=float(np.sqrt(np.mean((y - p) ** 2))), bias=float(p.mean() - y.mean()))


def region_transfer(df, arms, east, west):
    """E1: paper Table 6 (county corn, RF). Mean over seeds."""
    E, W = df[df.st.isin(east)], df[df.st.isin(west)]
    rows = []
    for arm, cols in arms.items():
        for name, (s, t) in {"East->West": (E, W), "West->East": (W, E)}.items():
            m = [metrics(t.y, rf(seed).fit(s[cols], s.y).predict(t[cols])) for seed in SEEDS]
            rows.append(dict(arm=arm, direction=name, **pd.DataFrame(m).mean().to_dict()))
    return pd.DataFrame(rows)


def oracle(frames, cols):
    out = {}
    for t in STATES:
        T = frames[t]
        r2s = []
        for seed in SEEDS:
            p = np.zeros(len(T))
            for tr, te in GroupKFold(5).split(T, groups=T.FIPS):
                p[te] = rf(seed).fit(T.iloc[tr][cols], T.iloc[tr].y).predict(T.iloc[te][cols])
            r2s.append(r2_score(T.y, p))
        out[t] = float(np.mean(r2s))
    return out


def pair_matrix(frames, cols):
    """E3: train in each state (all years), test in every other state (all years)."""
    acc = {}
    for seed in SEEDS:
        for s in STATES:
            S = frames[s]
            m = rf(seed).fit(S[cols], S.y)
            for t in STATES:
                if t != s:
                    acc.setdefault((s, t), []).append(metrics(frames[t].y, m.predict(frames[t][cols])))
    rows = [dict(src=s, tgt=t, **pd.DataFrame(v).mean().to_dict()) for (s, t), v in acc.items()]
    return pd.DataFrame(rows)
