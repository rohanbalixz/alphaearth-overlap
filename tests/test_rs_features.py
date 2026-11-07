import numpy as np
import pandas as pd
import pytest

from aeoverlap.rs_features import YVARS, build


@pytest.fixture
def row():
    """y = 0.3 + 0.2 cos(2 pi (t - t0)) for every band: peaks on day 200 at 0.5."""
    t0 = (200 - 1) / 365.25
    r = {"GEOID": "19195", "year": 2020}
    for yv in YVARS:
        r.update({f"c_{yv}": 0.3, f"cos1_{yv}": 0.2 * np.cos(2 * np.pi * t0),
                  f"sin1_{yv}": 0.2 * np.sin(2 * np.pi * t0), f"cos2_{yv}": 0.0, f"sin2_{yv}": 0.0})
    return r


def test_full_export_gives_80_features_with_correct_phenology(row):
    f = build(pd.DataFrame([row]))
    assert f.shape == (1, 82)
    assert abs(f.gcvi_peak[0] - 0.5) < 1e-4
    exp_b30 = 0.3 + 0.2 * np.cos(2 * np.pi * 30 / 365.25)
    assert abs(f.ndvi_b30[0] - exp_b30) < 1e-3 and abs(f.ndvi_a30[0] - exp_b30) < 1e-3
    assert abs(f.red_b30_int[0] - f.red_a30_int[0]) < 1e-6
    assert 30 * exp_b30 < f.red_b30_int[0] < 30 * 0.5


def test_lite_export_gives_20_features(row):
    lite = {k: v for k, v in row.items() if k in ("GEOID", "year") or k.endswith(("_ndvi", "_gcvi"))}
    f = build(pd.DataFrame([lite]))
    assert f.shape == (1, 22) and abs(f.gcvi_peak[0] - 0.5) < 1e-4
