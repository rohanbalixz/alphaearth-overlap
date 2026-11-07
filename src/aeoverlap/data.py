"""Data loading, input arms and study constants.

Source data: county-level corn table shared by Ma, Shen, Swatantran & Lobell (2026), IJAEOG,
https://github.com/yuchima8/Harvest_AlphaEarth (fetched and checksum-verified by
``scripts/get_data.sh``; not redistributed here).

Optional: ``data/rs_features.csv`` (Landsat harmonic features from ``gee/`` and
``aeoverlap.rs_features``). When present, the RS arms are added and every arm is restricted to the
county-years with complete RS features, so all arms are compared on identical rows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AEOVERLAP_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PREREG = RESULTS / "preregistered"        # E1-E6, fixed in PREREGISTRATION.md
EXPLORATORY = RESULTS / "exploratory"    # F1-F3, designed after the pre-registered results
FIGURES = RESULTS / "figures"
SOURCE_CSV = DATA / "US_FM_Corn_Data_shared.csv"

AEF = [f"A{k:02d}" for k in range(64)]
# Ma et al. (2026), Sec. 3.4: monthly GDD and PPT, May-Sep, for corn and soybean (10 predictors).
CLIM = [f"GDD_{m}" for m in range(5, 10)] + [f"ppt_{m}" for m in range(5, 10)]

ABBR = {"ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "MICHIGAN": "MI",
        "MINNESOTA": "MN", "MISSOURI": "MO", "NEBRASKA": "NE", "NORTH DAKOTA": "ND",
        "OHIO": "OH", "SOUTH DAKOTA": "SD", "WISCONSIN": "WI"}
# Ma et al. (2026), Sec. 3.5: EPA ecoregions. East = Eastern Temperate Forests, West = Great Plains.
EAST = ["IL", "IN", "MI", "OH", "WI"]
WEST = ["IA", "KS", "MN", "MO", "ND", "NE", "SD"]
STATES = sorted(ABBR.values())

N_TREES = 200          # Ma et al. (2026): random forest with 200 trees
SEEDS = range(5)       # Ma et al. (2026): each experiment repeated with five random seeds


def _ring_centroid(ring):
    x, y = np.asarray(ring, float).T
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = cross.sum() / 2
    if abs(a) < 1e-12:
        return x.mean(), y.mean(), 0.0
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * a)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * a)
    return cx, cy, abs(a)


def geo_centroid(geo_json: str) -> tuple[float, float]:
    """Area-weighted centroid (lon, lat) of a GeoJSON Polygon or MultiPolygon (outer rings)."""
    g = json.loads(geo_json)
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    parts = [_ring_centroid(p[0]) for p in polys]
    w = np.array([p[2] for p in parts])
    if w.sum() == 0:
        w = np.ones(len(parts))
    return (float(np.average([p[0] for p in parts], weights=w)),
            float(np.average([p[1] for p in parts], weights=w)))


def geo_area_km2(geo_json: str) -> float:
    """Approximate area (km^2) of a GeoJSON Polygon/MultiPolygon: outer rings, local equirectangular scale."""
    g = json.loads(geo_json)
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    total = 0.0
    for p in polys:
        lon, lat = np.asarray(p[0], float).T
        x, y = lon * 111.320 * np.cos(np.radians(lat.mean())), lat * 110.574
        total += abs((x[:-1] * y[1:] - x[1:] * y[:-1]).sum()) / 2
    return float(total)


USGS_CSV = DATA / "usgs" / "usco2015v2.0.csv"


def irrigation_share() -> pd.Series:
    """County irrigated area, 2015 (USGS; Dieter et al. 2018), in km^2, indexed by FIPS.
    Uses IR-IrTot (total irrigated thousand acres: crops + golf) because several states (NE, MO, ND, SD, WI)
    report only the total, not the crop split (IC-IrTot = "--"). Golf is negligible in farm counties.
    Divide by county area for an irrigation share. Used only in the exploratory follow-ups."""
    if not USGS_CSV.exists():
        raise FileNotFoundError(f"{USGS_CSV} not found; run `make data` first")
    u = pd.read_csv(USGS_CSV, skiprows=1, low_memory=False)
    u = u[pd.to_numeric(u["FIPS"], errors="coerce").notna()]
    irr_km2 = pd.to_numeric(u["IR-IrTot"], errors="coerce") * 1000 * 0.00404686
    return pd.Series(irr_km2.values, index=u["FIPS"].astype(int).values, name="irr_km2")


def load(use_rs: bool = True) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return ``(df, arms)``: one row per county-year, and arm name -> feature columns."""
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"{SOURCE_CSV} not found; run `make data` (scripts/get_data.sh) first")
    df = pd.read_csv(SOURCE_CSV)
    df["lon"], df["lat"] = zip(*df[".geo"].map(geo_centroid))
    df["area_km2"] = df[".geo"].map(geo_area_km2)
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
        arms["RS"] = rs_cols + CLIM          # the source study's RS baseline (80 Landsat + 10 climate)
        arms["RS-noclim"] = rs_cols          # Landsat only: the clean input-provenance contrast with AEF
    assert df[sum(arms.values(), [])].notna().all().all(), "missing feature values"
    return df, arms


def state_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: df[df.st == s].reset_index(drop=True) for s in STATES}
