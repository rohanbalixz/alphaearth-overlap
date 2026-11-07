import json

import numpy as np
import pandas as pd

from aeoverlap.data import geo_area_km2
from aeoverlap.run_exploratory import variance_shares


def test_area_of_one_degree_square_at_equator():
    sq = json.dumps({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]})
    assert abs(geo_area_km2(sq) - 111.320 * np.cos(np.radians(0.5)) * 110.574) < 1.0


def test_variance_shares_separate_location_from_time():
    r = np.random.default_rng(0)
    n_cty, years = 200, 8
    lon, lat = r.uniform(-100, -85, n_cty), r.uniform(37, 48, n_cty)
    rows = []
    for c in range(n_cty):
        for _ in range(years):
            rows.append(dict(FIPS=c, st=f"S{int(lon[c]) % 4}", lon=lon[c], lat=lat[c],
                             static=0.5 * lon[c] + lat[c],        # pure function of location
                             temporal=r.normal()))                 # pure year-to-year noise
    vs = variance_shares(pd.DataFrame(rows), ["static", "temporal"]).set_index("feature")
    assert vs.loc["static", "between_county"] > 0.999 and vs.loc["static", "location_surface"] > 0.99
    assert vs.loc["temporal", "between_county"] < 0.2 and vs.loc["temporal", "location_surface"] < 0.05
