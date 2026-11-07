"""Turn the Earth Engine export (data/gee/rs_harm_*.csv) into the paper's 80 RS features
(data/rs_features.csv): for each of 8 bands, 5 harmonic coefficients + peak, b30, a30, b30_int,
a30_int computed from the county's fitted curve (Ma et al. 2026, Sec. 3.4; Deines et al. 2021).

The paper defines Band_peak as "the maximum value in the growing season" for every band, and that
definition is followed literally here. Integrals are in reflectance x days; a random forest does not
depend on that scale.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

from common import DATA

YVARS = ["blue", "green", "red", "nir", "swir1", "swir2", "ndvi", "gcvi"]
XVARS = ["c", "cos1", "sin1", "cos2", "sin2"]
DOY0, DOY1 = 91, 305           # Apr 1 - Oct 31 (non-leap); window searched for the peak


def curve(coef, doy):
    """coef: (n, 5) in XVARS order; doy: (m,) -> (n, m)."""
    w = 2 * np.pi * (np.asarray(doy, float) - 1) / 365.25
    B = np.vstack([np.ones_like(w), np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)])
    return coef @ B


def phenology(coef):
    days = np.arange(DOY0, DOY1 + 1)
    yv = curve(coef, days)
    k = yv.argmax(axis=1)
    pk = days[k].astype(float)
    peak = yv[np.arange(len(yv)), k]
    out = dict(peak=peak,
               b30=np.array([curve(coef[i:i + 1], [pk[i] - 30])[0, 0] for i in range(len(pk))]),
               a30=np.array([curve(coef[i:i + 1], [pk[i] + 30])[0, 0] for i in range(len(pk))]))
    b_int, a_int = [], []
    for i in range(len(pk)):
        db, da = np.arange(pk[i] - 30, pk[i] + 1), np.arange(pk[i], pk[i] + 31)
        b_int.append(np.trapz(curve(coef[i:i + 1], db)[0], db))
        a_int.append(np.trapz(curve(coef[i:i + 1], da)[0], da))
    out["b30_int"], out["a30_int"] = np.array(b_int), np.array(a_int)
    return out


def build(raw):
    raw = raw.dropna(subset=[f"{x}_{y}" for x in XVARS for y in YVARS]).copy()
    feats = {"FIPS": raw["GEOID"].astype(int).values, "year": raw["year"].astype(int).values}
    for y in YVARS:
        coef = raw[[f"{x}_{y}" for x in XVARS]].values
        for x, col in zip(XVARS, coef.T):
            feats[f"{y}_{x}"] = col
        for name, v in phenology(coef).items():
            feats[f"{y}_{name}"] = v
    return pd.DataFrame(feats)


def main():
    files = sorted((DATA / "gee").glob("rs_harm_*.csv"))
    if not files:
        sys.exit("no data/gee/rs_harm_*.csv found; run gee/export_landsat_harmonics.py first")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).drop_duplicates(["GEOID", "year"])
    out = build(raw)
    assert out.shape[1] == 2 + 80, out.shape
    out.to_csv(DATA / "rs_features.csv", index=False)
    print(f"wrote data/rs_features.csv: {len(out)} county-years x 80 features "
          f"(dropped {len(raw) - len(out)} with no corn pixels / too few clear observations)")


if __name__ == "__main__":
    main()
