import numpy as np
import pandas as pd
import pytest

COLS = [f"f{i}" for i in range(8)]


def fake_state(n_cty, shift, d=8, years=6, seed=0):
    """Synthetic 'state': counties with a stable location in feature space plus yearly noise."""
    r = np.random.default_rng(seed)
    rows = []
    for c in range(n_cty):
        base = r.normal(shift, 1, d)
        for _ in range(years):
            x = base + r.normal(0, 0.3, d)
            rows.append(dict(FIPS=seed * 1000 + c, y=x[:3].sum() + r.normal(0, 0.5),
                             **{f"f{i}": x[i] for i in range(d)}))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def states():
    """A and B share a distribution; C is far away."""
    return fake_state(60, 0.0, seed=1), fake_state(60, 0.0, seed=2), fake_state(60, 8.0, seed=3)
