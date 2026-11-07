from conftest import COLS

from aeoverlap.overlap import full_space, haversine_km, knn_cover, yield_range_cover, yield_subspace


def test_full_space_separates_far_states_only(states):
    A, B, C = states
    same, far = full_space(A, B, COLS), full_space(A, C, COLS)
    assert same["auc"] < 0.65 and same["cover"] > 0.9
    assert far["auc"] > 0.99 and far["cover"] < 0.05


def test_yield_subspace_and_knn_agree_with_ground_truth(states):
    A, B, C = states
    assert yield_subspace(A, B, COLS)["cover"] > 0.9
    assert yield_subspace(A, C, COLS)["cover"] < 0.05
    assert knn_cover(A, B, COLS) > 0.8
    assert knn_cover(A, C, COLS) == 0.0


def test_yield_range_cover_of_itself(states):
    A, _, _ = states
    assert yield_range_cover(A, A) > 0.85


def test_haversine_one_degree_of_latitude():
    assert abs(haversine_km(0, 0, 0, 1) - 111.19) < 0.1
