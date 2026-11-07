"""Shared data loading and constants.

Data: county-level corn table from Ma, Shen, Swatantran & Lobell (2026), IJAEOG,
github.com/yuchima8/Harvest_AlphaEarth (run ./get_data.sh first).
Optional: data/rs_features.csv (Landsat harmonic features, built by
gee/export_landsat_harmonics.py + src/rs_features.py). When present, the RS arms are
added and every arm is restricted to the county-years that have complete RS features,
so all arms are compared on the same rows.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

AEF = [f"A{k:02d}" for k in range(64)]
# Paper Sec. 3.4: monthly GDD and PPT, May-Sep, for corn and soybean (10 predictors).
CLIM = [f"GDD_{m}" for m in range(5, 10)] + [f"ppt_{m}" for m in range(5, 10)]

ABBR = {"ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "MICHIGAN": "MI",
        "MINNESOTA": "MN", "MISSOURI": "MO", "NEBRASKA": "NE", "NORTH DAKOTA": "ND",
        "OHIO": "OH", "SOUTH DAKOTA": "SD", "WISCONSIN": "WI"}
# Paper Sec. 3.5: EPA ecoregions. East = Eastern Temperate Forests, West = Great Plains.
EAST = ["IL", "IN", "MI", "OH", "WI"]
WEST = ["IA", "KS", "MN", "MO", "ND", "NE", "SD"]
STATES = sorted(ABBR.values())

N_TREES = 200          # paper: RF with 200 trees
SEEDS = range(5)       # paper: each experiment repeated with five random seeds


def _ring_centroid(ring):
    x, y = np.asarray(ring, float).T
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = cross.sum() / 2
    if abs(a) < 1e-12:
        return x.mean(), y.mean(), 0.0
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * a)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * a)
    return cx, cy, abs(a)


def geo_centroid(geo_json):
    """Area-weighted centroid (lon, lat) of a GeoJSON Polygon/MultiPolygon (outer rings)."""
    g = json.loads(geo_json)
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    parts = [_ring_centroid(p[0]) for p in polys]
    w = np.array([p[2] for p in parts])
    if w.sum() == 0:
        w = np.ones(len(parts))
    return (float(np.average([p[0] for p in parts], weights=w)),
            float(np.average([p[1] for p in parts], weights=w)))


def load(use_rs=True):
    """Return (df, arms). df has one row per county-year; arms maps arm name -> feature list."""
    df = pd.read_csv(DATA / "US_FM_Corn_Data_shared.csv")
    df["lon"], df["lat"] = zip(*df[".geo"].map(geo_centroid))
    df = df.drop(columns=[".geo"])
    df["st"] = df["State"].map(ABBR)
    df["FIPS"] = df["FIPS"].astype(int)
    df["y"] = df["yield"].astype(float)          # bu/ac; R2 is unit-free
    arms = {"AEF": AEF, "AEF+clim": AEF + CLIM, "clim": CLIM}

    rs_path = DATA / "rs_features.csv"
    if use_rs and rs_path.exists():
        rs = pd.read_csv(rs_path)
        rs_cols = [c for c in rs.columns if c not in ("FIPS", "year")]
        df = df.merge(rs, on=["FIPS", "year"], how="left")
        n0 = len(df)
        df = df.dropna(subset=rs_cols).reset_index(drop=True)
        print(f"[load] RS features found: {len(rs_cols)} cols; kept {len(df)}/{n0} county-years with complete RS")
        arms["RS"] = rs_cols + CLIM          # the paper's RS baseline (80 Landsat + 10 climate)
        arms["RS-noclim"] = rs_cols          # Landsat only: the clean input-provenance contrast with AEF
    assert df[sum(arms.values(), [])].notna().all().all(), "missing feature values"
    return df, arms


def state_frames(df):
    return {s: df[df.st == s].reset_index(drop=True) for s in STATES}
