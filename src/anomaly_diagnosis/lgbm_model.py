from __future__ import annotations

from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def train_lgbm_cv(
    x_train: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    params: Dict,
    n_splits: int = 5,
    seed: int = 42,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
) -> Tuple[np.ndarray, np.ndarray, list[dict]]:
    classes = np.sort(pd.Series(y).unique())
    num_class = len(classes)

    model_params = dict(params)
    model_params["num_class"] = num_class
    model_params.setdefault("random_state", seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros((len(x_train), num_class), dtype=float)
    test_probs = np.zeros((len(x_test), num_class), dtype=float)
    metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y), start=1):
        x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_va, label=y_va, reference=dtrain)

        model = lgb.train(
            model_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        va_probs = model.predict(x_va, num_iteration=model.best_iteration)
        te_probs = model.predict(x_test, num_iteration=model.best_iteration)

        oof[va_idx] = va_probs
        test_probs += te_probs / n_splits

        va_pred = va_probs.argmax(axis=1)
        macro_f1 = f1_score(y_va, va_pred, average="macro")
        metrics.append(
            {
                "fold": fold,
                "macro_f1": float(macro_f1),
                "best_iteration": int(model.best_iteration or num_boost_round),
            }
        )

    return oof, test_probs, metrics
