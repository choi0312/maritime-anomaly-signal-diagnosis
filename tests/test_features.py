import numpy as np
import pandas as pd

from anomaly_diagnosis.features import FeatureEngineer


def make_data(n=80):
    rng = np.random.default_rng(42)
    x = pd.DataFrame(
        rng.normal(size=(n, 8)),
        columns=[f"X_{i:02d}" for i in range(1, 9)],
    )
    y = pd.Series(np.repeat(np.arange(4), n // 4))
    x["X_08"] = x["X_01"] * 0.9 + rng.normal(scale=0.01, size=n)
    return x, y


def test_feature_engineer_fit_transform_shape():
    x, y = make_data()
    fe = FeatureEngineer(
        corr_threshold=0.7,
        lda_components=3,
        kmeans_clusters=4,
        random_state=42,
    )
    transformed = fe.fit_transform(x, y)

    assert len(transformed) == len(x)
    assert transformed.shape[1] > x.shape[1]
    assert np.isfinite(transformed.values).all()


def test_feature_engineer_transform_consistency():
    x, y = make_data()
    fe = FeatureEngineer(kmeans_clusters=4)
    train_out = fe.fit_transform(x, y)
    test_out = fe.transform(x.iloc[:10])

    assert list(train_out.columns) == list(test_out.columns)
    assert len(test_out) == 10
