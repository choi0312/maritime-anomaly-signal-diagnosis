import numpy as np
import pandas as pd

from anomaly_diagnosis.postprocess import apply_switch_rule, balance_two_labels


def test_switch_rule_positive_and_negative_label():
    probs = np.ones((3, 20)) / 20
    raw = pd.DataFrame(
        {
            "X_16": [0.9, 0.1, 0.1],
            "X_18": [0.1, 0.1, 0.1],
            "X_26": [0.5, 0.5, 0.1],
            "X_30": [0.5, 0.5, 0.5],
        }
    )
    labels = apply_switch_rule(probs, raw)
    assert labels[0] == 2
    assert labels[2] == 6


def test_balance_two_labels_reduces_gap():
    labels = np.array([0, 0, 0, 0, 15, 1])
    probs = np.zeros((6, 20))
    probs[:, 0] = 0.6
    probs[:, 15] = 0.4
    probs[0, 0] = 0.51
    probs[0, 15] = 0.49

    out = balance_two_labels(labels, probs, label_a=0, label_b=15)
    assert abs((out == 0).sum() - (out == 15).sum()) < abs(
        (labels == 0).sum() - (labels == 15).sum()
    )
