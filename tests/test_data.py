import json

import numpy as np

from aeoverlap.data import AEF, CLIM, EAST, STATES, WEST, geo_centroid


def test_centroids():
    sq = json.dumps({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]})
    assert np.allclose(geo_centroid(sq), (0.5, 0.5))
    mp = json.dumps({"type": "MultiPolygon", "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                                                            [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]]]})
    assert np.allclose(geo_centroid(mp), (1.5, 0.5))


def test_study_constants_match_the_source_paper():
    assert len(AEF) == 64 and len(CLIM) == 10
    assert sorted(EAST + WEST) == STATES and len(STATES) == 12
