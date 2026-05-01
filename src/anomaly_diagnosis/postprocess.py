from __future__ import annotations

import numpy as np
import pandas as pd


def apply_switch_rule(
    probs: np.ndarray,
    raw_test: pd.DataFrame,
    high_threshold: float = 0.8,
    low_threshold: float = 0.3,
    pos_cols: list[str] | None = None,
    neg_cols: list[str] | None = None,
    pos_label: int = 2,
    neg_label: int = 6,
) -> np.ndarray:
    pos_cols = pos_cols or ["X_16", "X_18"]
    neg_cols = neg_cols or ["X_26", "X_30"]

    final = np.argmax(probs, axis=1).astype(int)

    for i, row in raw_test.reset_index(drop=True).iterrows():
        pos_trigger = any(col in raw_test.columns and row[col] >= high_threshold for col in pos_cols)
        neg_trigger = any(col in raw_test.columns and row[col] <= low_threshold for col in neg_cols)

        if pos_trigger and neg_trigger:
            final[i] = pos_label if probs[i, pos_label] >= probs[i, neg_label] else neg_label
        elif pos_trigger:
            final[i] = pos_label
        elif neg_trigger:
            final[i] = neg_label
        else:
            if final[i] in {pos_label, neg_label}:
                adjusted = probs[i].copy()
                adjusted[pos_label] = -np.inf
                adjusted[neg_label] = -np.inf
                final[i] = int(np.argmax(adjusted))

    return final


def balance_two_labels(
    labels: np.ndarray,
    probs: np.ndarray,
    label_a: int = 0,
    label_b: int = 15,
) -> np.ndarray:
    labels = labels.copy()
    idx = np.where(np.isin(labels, [label_a, label_b]))[0]

    if len(idx) == 0:
        return labels

    count_a = int(np.sum(labels[idx] == label_a))
    count_b = int(np.sum(labels[idx] == label_b))

    diff = abs(count_a - count_b)
    if diff <= 1:
        return labels

    majority = label_a if count_a > count_b else label_b
    minority = label_b if majority == label_a else label_a
    n_flip = diff // 2

    candidates = idx[labels[idx] == majority]
    margins = probs[candidates, majority] - probs[candidates, minority]
    flip_idx = candidates[np.argsort(margins)[:n_flip]]

    labels[flip_idx] = minority
    return labels


def apply_postprocessing(
    probs: np.ndarray,
    raw_test: pd.DataFrame,
    config: dict,
) -> np.ndarray:
    labels = np.argmax(probs, axis=1).astype(int)

    switch_cfg = config.get("switch_rule", {})
    if switch_cfg.get("enabled", True):
        labels = apply_switch_rule(
            probs=probs,
            raw_test=raw_test,
            high_threshold=float(switch_cfg.get("high_threshold", 0.8)),
            low_threshold=float(switch_cfg.get("low_threshold", 0.3)),
            pos_cols=switch_cfg.get("pos_cols", ["X_16", "X_18"]),
            neg_cols=switch_cfg.get("neg_cols", ["X_26", "X_30"]),
            pos_label=int(switch_cfg.get("pos_label", 2)),
            neg_label=int(switch_cfg.get("neg_label", 6)),
        )

    balance_cfg = config.get("balance_rule", {})
    if balance_cfg.get("enabled", True):
        labels = balance_two_labels(
            labels=labels,
            probs=probs,
            label_a=int(balance_cfg.get("label_a", 0)),
            label_b=int(balance_cfg.get("label_b", 15)),
        )

    return labels
