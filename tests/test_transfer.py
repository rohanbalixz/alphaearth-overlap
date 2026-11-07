import numpy as np

from aeoverlap.transfer import metrics


def test_level_fix_removes_a_pure_offset_exactly():
    y = np.random.default_rng(0).normal(0, 1, 500)
    m = metrics(y, y + 3.0)
    assert m["R2"] < 0
    assert abs(m["R2_level"] - 1) < 1e-12
    assert abs(m["bias"] - 3) < 1e-12 and abs(m["rmse"] - 3) < 1e-12


def test_ranking_error_survives_level_fix():
    r = np.random.default_rng(1)
    y = r.normal(0, 1, 500)
    p = 0.3 * y + r.normal(0, 1, 500) + 2.0
    m = metrics(y, p)
    assert m["R2_level"] < 0.5 and m["R2_level"] > m["R2"]
