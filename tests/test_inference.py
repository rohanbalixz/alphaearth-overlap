import numpy as np
import pandas as pd
import pytest

from aeoverlap.data import STATES
from aeoverlap.inference import dsp_partial, qap_spearman


@pytest.fixture(scope="module")
def pairs():
    rng = np.random.default_rng(0)
    p = pd.DataFrame([(s, t) for s in STATES for t in STATES if s != t], columns=["src", "tgt"])
    x = rng.normal(size=len(p))
    return p.assign(x=x, y_rel=-x + rng.normal(0, 0.5, len(p)), c=rng.normal(size=len(p)))


def test_qap_detects_planted_relation(pairs):
    rho, p = qap_spearman(pairs, "x", "y_rel", 999)
    assert rho < -0.7 and p < 0.01


def test_qap_size_under_the_null(pairs):
    def null_p(i):
        y0 = np.random.default_rng(i).normal(size=len(pairs))
        return qap_spearman(pairs.assign(y0=y0), "x", "y0", 199, seed=i)[1]

    rejections = sum(null_p(i) < 0.05 for i in range(60))
    assert rejections <= 9          # ~3 expected at the 5% level


def test_dsp_partial_with_fixed_effects(pairs):
    coef, t, p = dsp_partial(pairs, "x", "y_rel", ["c"], fe=True, n_perm=499)
    assert coef < -0.5 and p < 0.01
