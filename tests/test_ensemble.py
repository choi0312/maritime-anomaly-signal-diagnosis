import numpy as np

from anomaly_diagnosis.ensemble import blend_probs, row_normalize


def test_row_normalize():
    arr = np.array([[2.0, 2.0], [1.0, 3.0]])
    out = row_normalize(arr)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_blend_probs_shape():
    a = np.ones((5, 3)) / 3
    b = np.ones((5, 3)) / 3
    out = blend_probs(a, b, weight_a=0.6)
    assert out.shape == (5, 3)
    assert np.allclose(out.sum(axis=1), 1.0)
