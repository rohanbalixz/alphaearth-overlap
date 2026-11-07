"""Export county-mean Landsat harmonic coefficients over CDL corn pixels, 2017-2024 (the RS arm).

Rebuilds the RS features of Ma, Shen, Swatantran & Lobell (2026), Sec. 3.2-3.4, for county corn:
Landsat 7/8/9 Collection 2 Level-2 surface reflectance; six bands (blue, green, red, NIR, SWIR1,
SWIR2) + NDVI + GCVI; QA mask (cloud, dilated cloud, cloud shadow, snow, water) and saturation
mask; CDL corn mask; second-order harmonic regression
    y(t) = c + a1 cos(2 pi t) + b1 sin(2 pi t) + a2 cos(4 pi t) + b2 sin(4 pi t),  t = fraction of year.
src/rs_features.py then derives the phenology metrics (peak, b30, a30, b30_int, a30_int) from the
county curves, giving 10 features x 8 bands = 80, as in the paper.

Documented deviations (see README):
  1. Harmonics are fitted per pixel and then averaged over corn pixels in each county; the paper
     averages per date and then fits. OLS coefficients are linear in y, so the two agree exactly when
     pixels share observation dates and approximately otherwise. This is far cheaper on Earth Engine.
  2. Growing-season window: the paper does not state it. Default Apr 1 - Oct 31 (--start/--end).
  3. GCVI = NIR/Green - 1 (the standard form). A constant offset does not change a random forest.

UNTESTED on Earth Engine: written on a machine without EE credentials. It must pass the gate in
README before its features are trusted: RS arm East->West R2 ~0.66, West->East ~0.60 (paper Table 6).

Usage:
  pip install earthengine-api && earthengine authenticate
  python gee/export_landsat_harmonics.py --project <gcp-project>               # 8 tasks (one per year)
  python gee/export_landsat_harmonics.py --project <gcp-project> --per-state   # 96 smaller tasks
  python gee/export_landsat_harmonics.py --project <gcp-project> --lite        # low compute: GCVI+NDVI, 120 m
  python gee/export_landsat_harmonics.py --project <gcp-project> --lite --years 2017   # measure cost first
  # Tasks write CSVs to Google Drive folder alphaearth_overlap/. Put them in data/gee/, then:
  .venv/bin/python -m aeoverlap.rs_features
"""
import argparse
import math
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import ee  # noqa: E402

STATEFP = {"IL": "17", "IN": "18", "IA": "19", "KS": "20", "MI": "26", "MN": "27",
           "MO": "29", "NE": "31", "ND": "38", "OH": "39", "SD": "46", "WI": "55"}
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
YVARS = BANDS + ["ndvi", "gcvi"]
XVARS = ["c", "cos1", "sin1", "cos2", "sin2"]
SOURCES = {  # collection id -> SR band names in BANDS order
    "LANDSAT/LE07/C02/T1_L2": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
    "LANDSAT/LC08/C02/T1_L2": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
    "LANDSAT/LC09/C02/T1_L2": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
}
# Bounding box of the 12 states (ND/NE/KS western edge to OH eastern edge).
BBOX = [-104.1, 35.9, -80.4, 49.1]
QA_BAD = (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 7)  # dilated cloud, cloud, shadow, snow, water


def prep(img, sr_bands):
    img = ee.Image(img)
    clear = img.select("QA_PIXEL").bitwiseAnd(QA_BAD).eq(0).And(img.select("QA_RADSAT").eq(0))
    sr = img.select(sr_bands, BANDS).multiply(2.75e-05).add(-0.2)
    valid = sr.reduce(ee.Reducer.min()).gt(0).And(sr.reduce(ee.Reducer.max()).lt(1))
    ndvi = sr.normalizedDifference(["nir", "red"]).rename("ndvi")
    gcvi = sr.select("nir").divide(sr.select("green")).subtract(1).rename("gcvi")
    d = img.date()
    w = ee.Number(d.difference(ee.Date.fromYMD(d.get("year"), 1, 1), "year")).multiply(2 * math.pi)
    harm = ee.Image.cat([ee.Image.constant(1), ee.Image.constant(w.cos()), ee.Image.constant(w.sin()),
                         ee.Image.constant(w.multiply(2).cos()), ee.Image.constant(w.multiply(2).sin())]
                        ).rename(XVARS)
    out = harm.addBands(sr).addBands(ndvi).addBands(gcvi).float().updateMask(clear.And(valid))
    return ee.Image(out.copyProperties(img, ["system:time_start"]))


def county_coefficients(year, counties, start, end, min_obs, yvars=YVARS, scale=30):
    region = ee.Geometry.Rectangle(BBOX)
    coll = None
    for cid, bands in SOURCES.items():
        c = (ee.ImageCollection(cid).filterBounds(region)
             .filterDate(f"{year}-{start}", f"{year}-{end}")
             .map(lambda im, b=bands: prep(im, b)))
        coll = c if coll is None else coll.merge(c)
    coll = coll.select(XVARS + yvars)
    fit = coll.reduce(ee.Reducer.linearRegression(numX=len(XVARS), numY=len(yvars)))
    coef = fit.select("coefficients").arrayFlatten([XVARS, yvars])      # bands like "cos1_gcvi"
    nobs = coll.select(yvars[0]).count().rename("nobs")
    corn = ee.Image(f"USDA/NASS/CDL/{year}").select("cropland").eq(1)
    img = coef.addBands(nobs).updateMask(nobs.gte(min_obs)).updateMask(corn).float()
    stats = img.reduceRegions(collection=counties, reducer=ee.Reducer.mean(), scale=scale, tileScale=16)
    return stats.map(lambda f: ee.Feature(None, f.toDictionary()).set("year", year))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2017, 2025)))
    ap.add_argument("--start", default="04-01", help="growing-season start MM-DD (inclusive)")
    ap.add_argument("--end", default="11-01", help="growing-season end MM-DD (exclusive)")
    ap.add_argument("--min-obs", type=int, default=8, help="min clear observations per pixel for a fit")
    ap.add_argument("--per-state", action="store_true", help="one task per state-year instead of per year")
    ap.add_argument("--drive-folder", default="alphaearth_overlap")
    ap.add_argument("--lite", action="store_true",
                    help="low-compute version: GCVI + NDVI only, county means at 120 m (PREREGISTRATION Amendment 1)")
    ap.add_argument("--scale", type=int, default=None, help="reduceRegions scale in m (default 30; 120 with --lite)")
    ap.add_argument("--smoke", metavar="GEOID", default=None,
                    help="compute ONE county for the first --years value interactively (no export), print and check it")
    a = ap.parse_args()
    yvars = ["ndvi", "gcvi"] if a.lite else YVARS
    scale = a.scale or (120 if a.lite else 30)
    tag = "rs_lite" if a.lite else "rs_harm"

    ee.Initialize(project=a.project)
    all_cty = ee.FeatureCollection("TIGER/2018/Counties").filter(
        ee.Filter.inList("STATEFP", list(STATEFP.values())))
    groups = {k: [v] for k, v in STATEFP.items()} if a.per_state else {"all": list(STATEFP.values())}
    selectors = ["GEOID", "STATEFP", "year", "nobs"] + [f"{x}_{y}" for x in XVARS for y in yvars]
    print(f"bands {yvars}, scale {scale} m, years {a.years}")

    if a.smoke:
        cty = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq("GEOID", a.smoke))
        fc = county_coefficients(a.years[0], cty, a.start, a.end, a.min_obs, yvars, scale)
        props = fc.first().getInfo()["properties"]
        for k in selectors:
            v = props.get(k)
            print(f"  {k:12s} {v:.4f}" if isinstance(v, float) else f"  {k:12s} {v}")
        missing = [k for k in selectors if props.get(k) is None]
        if missing:
            raise SystemExit(f"SMOKE FAILED: missing {missing}")
        # the fitted GCVI curve for a corn county should peak in summer at a plausible level
        import numpy as np
        days = np.arange(91, 306)
        w = 2 * np.pi * (days - 1) / 365.25
        g = (props["c_gcvi"] + props["cos1_gcvi"] * np.cos(w) + props["sin1_gcvi"] * np.sin(w)
             + props["cos2_gcvi"] * np.cos(2 * w) + props["sin2_gcvi"] * np.sin(2 * w))
        print(f"  fitted GCVI peak {g.max():.2f} on day {days[g.argmax()]}; "
              f"mean clear obs per pixel {props['nobs']:.1f}")
        ok = 150 <= days[g.argmax()] <= 250 and 1.5 <= g.max() <= 12 and props["nobs"] >= a.min_obs
        print("SMOKE OK" if ok else "SMOKE SUSPICIOUS: check the values above before exporting")
        return
    for year in a.years:
        for name, fps in groups.items():
            cty = all_cty.filter(ee.Filter.inList("STATEFP", fps))
            desc = f"{tag}_{year}" + ("" if name == "all" else f"_{name}")
            task = ee.batch.Export.table.toDrive(
                collection=county_coefficients(year, cty, a.start, a.end, a.min_obs, yvars, scale),
                description=desc, folder=a.drive_folder, fileNamePrefix=desc,
                fileFormat="CSV", selectors=selectors)
            task.start()
            print("started", desc, task.id)
    print("Monitor at https://code.earthengine.google.com/tasks or with: earthengine task list")


if __name__ == "__main__":
    main()
