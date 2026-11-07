"""Sanity checks on synthetic data with known answers. Run: .venv/bin/python tests/test_units.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from common import STATES, geo_centroid
from overlap import full_space, knn_cover, yield_range_cover, yield_subspace
from stats import dsp_partial, qap_spearman
from transfer import metrics

rng = np.random.default_rng(0)


def fake_state(n_cty, shift, d=8, years=6, seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for c in range(n_cty):
        base = r.normal(shift, 1, d)
        for _ in range(years):
            x = base + r.normal(0, 0.3, d)
            rows.append(dict(FIPS=seed * 1000 + c, y=x[:3].sum() + r.normal(0, 0.5), **{f"f{i}": x[i] for i in range(d)}))
    return pd.DataFrame(rows)


cols = [f"f{i}" for i in range(8)]
A, B, C = fake_state(60, 0.0, seed=1), fake_state(60, 0.0, seed=2), fake_state(60, 8.0, seed=3)

# 1. same distribution -> high overlap; far-apart -> none
same, far = full_space(A, B, cols), full_space(A, C, cols)
assert same["auc"] < 0.65 and same["cover"] > 0.9, same
assert far["auc"] > 0.99 and far["cover"] < 0.05, far
assert yield_subspace(A, B, cols)["cover"] > 0.9 and yield_subspace(A, C, cols)["cover"] < 0.05
assert knn_cover(A, B, cols) > 0.8 and knn_cover(A, C, cols) == 0.0
assert yield_range_cover(A, A) > 0.85
print("overlap measures OK:", {k: round(v, 3) for k, v in same.items()}, {k: round(v, 3) for k, v in far.items()})

# 2. level fix removes a pure offset exactly; ranking errors remain
y = rng.normal(0, 1, 500)
m = metrics(y, y + 3.0)
assert m["R2"] < 0 and abs(m["R2_level"] - 1) < 1e-12 and abs(m["bias"] - 3) < 1e-12
print("level decomposition OK:", {k: round(v, 3) for k, v in m.items()})

# 3. QAP: strong planted relation is detected; a pure state-level artefact is not over-rejected
pairs = pd.DataFrame([(s, t) for s in STATES for t in STATES if s != t], columns=["src", "tgt"])
x = rng.normal(size=len(pairs))
pairs["x"], pairs["y_rel"], pairs["y_null"] = x, -x + rng.normal(0, 0.5, len(pairs)), rng.normal(size=len(pairs))
rho, p = qap_spearman(pairs, "x", "y_rel", 999)
assert rho < -0.7 and p < 0.01, (rho, p)
rejections = sum(qap_spearman(pairs.assign(y_null=np.random.default_rng(i).normal(size=len(pairs))), "x", "y_null", 199, seed=i)[1] < 0.05
                 for i in range(60))
assert rejections <= 9, rejections            # ~5% expected of 60
coef, t, pd_ = dsp_partial(pairs.assign(c=rng.normal(size=len(pairs))), "x", "y_rel", ["c"], fe=True, n_perm=499)
assert coef < -0.5 and pd_ < 0.01, (coef, pd_)
print(f"QAP OK: planted rho={rho:.2f} p={p:.4f}; null rejections {rejections}/60; DSP coef={coef:.2f} p={pd_:.4f}")

# 4. centroid of a unit square and a multipolygon
sq = json.dumps({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]})
assert np.allclose(geo_centroid(sq), (0.5, 0.5))
mp = json.dumps({"type": "MultiPolygon", "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                                                        [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]]]})
assert np.allclose(geo_centroid(mp), (1.5, 0.5))
print("centroids OK")
print("ALL TESTS PASSED")

# 5. RS phenology: a curve y = 0.3 + 0.2 cos(2 pi (t - t0)) peaks at doy 200 with value 0.5
from rs_features import XVARS, YVARS, build, curve
t0 = (200 - 1) / 365.25
row = {"GEOID": "19195", "year": 2020}
for yv in YVARS:
    row.update({f"c_{yv}": 0.3, f"cos1_{yv}": 0.2 * np.cos(2 * np.pi * t0), f"sin1_{yv}": 0.2 * np.sin(2 * np.pi * t0),
                f"cos2_{yv}": 0.0, f"sin2_{yv}": 0.0})
f = build(pd.DataFrame([row]))
assert f.shape == (1, 82), f.shape
assert abs(f.gcvi_peak[0] - 0.5) < 1e-4, f.gcvi_peak[0]
exp_b30 = 0.3 + 0.2 * np.cos(2 * np.pi * 30 / 365.25)
assert abs(f.ndvi_b30[0] - exp_b30) < 1e-3 and abs(f.ndvi_a30[0] - exp_b30) < 1e-3
assert abs(f.red_b30_int[0] - f.red_a30_int[0]) < 1e-6 and 30 * exp_b30 < f.red_b30_int[0] < 30 * 0.5
print("RS phenology OK: peak", round(f.gcvi_peak[0], 4), "b30", round(f.ndvi_b30[0], 4), "int", round(f.red_b30_int[0], 3))
print("ALL TESTS PASSED (incl. RS)")
