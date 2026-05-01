from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def row_normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, None)
    return probs / probs.sum(axis=1, keepdims=True)


def blend_probs(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    weight_a: float = 0.575,
) -> np.ndarray:
    if probs_a.shape != probs_b.shape:
        raise ValueError(f"Probability shape mismatch: {probs_a.shape} vs {probs_b.shape}")

    weight_b = 1.0 - weight_a
    blended = weight_a * row_normalize(probs_a) + weight_b * row_normalize(probs_b)
    return row_normalize(blended)


def macro_f1_from_probs(y_true, probs: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)
    return float(f1_score(y_true, pred, average="macro"))
