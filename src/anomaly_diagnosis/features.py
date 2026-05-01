from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import pinv
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


CLEAR_SHIFT_SIGN: Dict[str, int] = {
    "X_01": +1, "X_02": +1, "X_03": -1, "X_04": -1, "X_05": +1, "X_06": -1,
    "X_07": -1, "X_08": +1, "X_09": -1, "X_10": -1, "X_11": +1, "X_12": -1,
    "X_13": -1, "X_14": +1, "X_15": +1, "X_16": +1, "X_17": -1, "X_18": +1,
    "X_19": +1, "X_20": -1, "X_21": -1, "X_22": -1, "X_23": -1, "X_24": -1,
    "X_25": +1, "X_26": -1, "X_27": -1, "X_28": +1, "X_29": -1, "X_30": -1,
    "X_31": -1, "X_32": -1, "X_33": -1, "X_34": +1, "X_35": +1, "X_36": +1,
    "X_37": +1, "X_38": +1, "X_39": -1, "X_40": +1, "X_41": -1, "X_42": +1,
    "X_43": +1, "X_44": +1, "X_45": -1, "X_46": -1, "X_47": +1, "X_48": -1,
    "X_49": -1, "X_50": -1, "X_51": +1, "X_52": -1,
}


def _safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _num_id(colname: str) -> str:
    found = re.findall(r"\d+", str(colname))
    return found[-1].zfill(2) if found else str(colname)


def _rowwise_mahalanobis(values: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    diff = values - mean.reshape(1, -1)
    dist_sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return np.sqrt(np.maximum(dist_sq, 0.0))


@dataclass
class FeatureEngineer:
    corr_threshold: float = 0.70
    mahalanobis_ridge: float = 1e-6
    lda_components: int = 10
    kmeans_clusters: int = 20
    add_corr_pairs: bool = True
    add_mahalanobis: bool = True
    add_clear_shift: bool = True
    add_lda: bool = True
    add_kmeans_distance: bool = True
    random_state: int = 42

    base_columns_: List[str] = field(default_factory=list)
    classes_: np.ndarray | None = None
    corr_pairs_: List[Tuple[str, str, float]] = field(default_factory=list)
    mahalanobis_stats_: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    scaler_: Optional[StandardScaler] = None
    lda_: Optional[LinearDiscriminantAnalysis] = None
    kmeans_scaler_: Optional[StandardScaler] = None
    kmeans_: Optional[KMeans] = None
    clear_mean_: Optional[pd.Series] = None
    clear_std_: Optional[pd.Series] = None
    clear_q05_: Optional[pd.Series] = None
    clear_q95_: Optional[pd.Series] = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "FeatureEngineer":
        x = _safe_numeric(x)
        self.base_columns_ = list(x.columns)
        self.classes_ = np.sort(pd.Series(y).unique())

        if self.add_corr_pairs:
            self._fit_corr_pairs(x)

        if self.add_mahalanobis:
            self._fit_mahalanobis(x, y)

        if self.add_clear_shift:
            self._fit_clear_shift(x)

        base = self._make_pre_projection_features(x).reset_index(drop=True)

        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(base)

        if self.add_lda:
            n_classes = len(self.classes_)
            max_components = min(self.lda_components, n_classes - 1, scaled.shape[1])
            if max_components >= 1:
                self.lda_ = LinearDiscriminantAnalysis(n_components=max_components)
                self.lda_.fit(scaled, y)

        if self.add_kmeans_distance:
            self.kmeans_scaler_ = StandardScaler()
            scaled_for_kmeans = self.kmeans_scaler_.fit_transform(base)
            n_clusters = max(2, min(self.kmeans_clusters, len(x)))
            self.kmeans_ = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            self.kmeans_.fit(scaled_for_kmeans)

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = _safe_numeric(x)

        missing = [c for c in self.base_columns_ if c not in x.columns]
        if missing:
            raise ValueError(f"Missing columns at transform time: {missing[:5]}")

        x = x[self.base_columns_].copy()
        base = self._make_pre_projection_features(x).reset_index(drop=True)
        outputs = [base]

        if self.scaler_ is not None and self.lda_ is not None:
            scaled = self.scaler_.transform(base)
            lda_values = self.lda_.transform(scaled)
            lda_df = pd.DataFrame(
                lda_values,
                columns=[f"lda_{i:02d}" for i in range(lda_values.shape[1])],
            )
            outputs.append(lda_df)

        if self.kmeans_ is not None and self.kmeans_scaler_ is not None:
            scaled_k = self.kmeans_scaler_.transform(base)
            distances = self.kmeans_.transform(scaled_k)
            dist_df = pd.DataFrame(
                distances,
                columns=[f"kmeans_dist_{i:02d}" for i in range(distances.shape[1])],
            )
            outputs.append(dist_df)

        result = pd.concat(outputs, axis=1)
        return _safe_numeric(result)

    def fit_transform(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(x, y).transform(x)

    def _fit_corr_pairs(self, x: pd.DataFrame) -> None:
        corr = x.corr(numeric_only=True)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().reset_index()
        pairs.columns = ["feature1", "feature2", "corr"]
        pairs = pairs[pairs["corr"].abs() >= self.corr_threshold]

        self.corr_pairs_ = [
            (str(row.feature1), str(row.feature2), float(row.corr))
            for row in pairs.itertuples(index=False)
        ]

    def _fit_mahalanobis(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.mahalanobis_stats_.clear()
        values = x.values

        for label in self.classes_:
            idx = np.where(np.asarray(y) == label)[0]
            class_values = values[idx]

            if len(class_values) < 2:
                mean = values.mean(axis=0)
                cov = np.cov(values, rowvar=False)
            else:
                mean = class_values.mean(axis=0)
                cov = np.cov(class_values, rowvar=False)

            cov = np.atleast_2d(cov)
            if cov.shape[0] != x.shape[1]:
                cov = np.eye(x.shape[1])

            cov = cov + np.eye(cov.shape[0]) * self.mahalanobis_ridge
            inv_cov = pinv(cov)
            self.mahalanobis_stats_[int(label)] = (mean, inv_cov)

    def _fit_clear_shift(self, x: pd.DataFrame) -> None:
        self.clear_mean_ = x.mean(axis=0)
        self.clear_std_ = x.std(axis=0).replace(0, 1e-6)
        self.clear_q05_ = x.quantile(0.05)
        self.clear_q95_ = x.quantile(0.95)

    def _make_pre_projection_features(self, x: pd.DataFrame) -> pd.DataFrame:
        parts = [x.reset_index(drop=True)]

        if self.add_corr_pairs:
            parts.append(self._make_corr_pair_features(x).reset_index(drop=True))

        if self.add_mahalanobis:
            parts.append(self._make_mahalanobis_features(x).reset_index(drop=True))

        if self.add_clear_shift:
            parts.append(self._make_clear_shift_features(x).reset_index(drop=True))

        return pd.concat(parts, axis=1)

    def _make_corr_pair_features(self, x: pd.DataFrame) -> pd.DataFrame:
        data = {}

        for f1, f2, corr in self.corr_pairs_:
            id1, id2 = _num_id(f1), _num_id(f2)
            if corr >= 0:
                data[f"corr_{id1}_{id2}_sum"] = x[f1].values + x[f2].values
            else:
                data[f"corr_{id1}_{id2}_diff"] = x[f1].values - x[f2].values

        return pd.DataFrame(data, index=x.index)

    def _make_mahalanobis_features(self, x: pd.DataFrame) -> pd.DataFrame:
        values = x.values
        data = {}

        for label in self.classes_:
            mean, inv_cov = self.mahalanobis_stats_[int(label)]
            data[f"m_dist_c{int(label)}"] = _rowwise_mahalanobis(values, mean, inv_cov)

        return pd.DataFrame(data, index=x.index)

    def _make_clear_shift_features(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.clear_mean_ is None or self.clear_std_ is None:
            raise RuntimeError("Clear-shift statistics are not fitted.")

        data = {}
        z = (x - self.clear_mean_) / self.clear_std_
        z = z.clip(-10, 10)

        for col in x.columns:
            sign = CLEAR_SHIFT_SIGN.get(col, 1)
            aligned = sign * z[col]
            data[f"{col}_align"] = aligned.values
            data[f"{col}_abs"] = np.abs(z[col].values)
            data[f"{col}_sq"] = np.square(z[col].values)
            data[f"{col}_low_extreme"] = (x[col] <= self.clear_q05_[col]).astype(int).values
            data[f"{col}_high_extreme"] = (x[col] >= self.clear_q95_[col]).astype(int).values

        return pd.DataFrame(data, index=x.index)
