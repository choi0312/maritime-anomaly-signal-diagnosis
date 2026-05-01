from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from anomaly_diagnosis.config import load_config
from anomaly_diagnosis.data import read_csv, split_xy
from anomaly_diagnosis.ensemble import blend_probs, macro_f1_from_probs
from anomaly_diagnosis.features import FeatureEngineer
from anomaly_diagnosis.lgbm_model import train_lgbm_cv
from anomaly_diagnosis.postprocess import apply_postprocessing
from anomaly_diagnosis.train_torch import train_torch_cv
from anomaly_diagnosis.utils import ensure_dir, save_json, seed_everything


def run_pipeline(config_path: str | Path) -> dict:
    cfg = load_config(config_path)

    seed = int(cfg["project"].get("seed", 42))
    seed_everything(seed)

    output_dir = ensure_dir(cfg["project"]["output_dir"])

    train_df = read_csv(cfg["data"]["train_path"])
    test_df = read_csv(cfg["data"]["test_path"])
    sample_sub = read_csv(cfg["data"]["sample_submission_path"])

    id_col = cfg["data"].get("id_col", "ID")
    target_col = cfg["data"].get("target_col", "target")

    x_train_raw, y, x_test_raw = split_xy(
        train_df=train_df,
        test_df=test_df,
        id_col=id_col,
        target_col=target_col,
    )

    fe_cfg = cfg["features"]
    engineer = FeatureEngineer(
        corr_threshold=float(fe_cfg.get("corr_threshold", 0.70)),
        mahalanobis_ridge=float(fe_cfg.get("mahalanobis_ridge", 1e-6)),
        lda_components=int(fe_cfg.get("lda_components", 10)),
        kmeans_clusters=int(fe_cfg.get("kmeans_clusters", 20)),
        add_corr_pairs=bool(fe_cfg.get("add_corr_pairs", True)),
        add_mahalanobis=bool(fe_cfg.get("add_mahalanobis", True)),
        add_clear_shift=bool(fe_cfg.get("add_clear_shift", True)),
        add_lda=bool(fe_cfg.get("add_lda", True)),
        add_kmeans_distance=bool(fe_cfg.get("add_kmeans_distance", True)),
        random_state=seed,
    )

    x_train = engineer.fit_transform(x_train_raw, y)
    x_test = engineer.transform(x_test_raw)

    joblib.dump(engineer, output_dir / "feature_engineer.joblib")

    n_splits = int(cfg["validation"].get("n_splits", 5))

    oof_collection = []
    test_collection = []
    metrics = {
        "feature_count": int(x_train.shape[1]),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "models": {},
    }

    if cfg["lightgbm"].get("enabled", True):
        oof_lgbm, test_lgbm, lgbm_metrics = train_lgbm_cv(
            x_train=x_train,
            y=y,
            x_test=x_test,
            params=cfg["lightgbm"]["params"],
            n_splits=n_splits,
            seed=seed,
            num_boost_round=int(cfg["lightgbm"].get("num_boost_round", 500)),
            early_stopping_rounds=int(cfg["lightgbm"].get("early_stopping_rounds", 50)),
        )
        np.save(output_dir / "oof_lgbm.npy", oof_lgbm)
        np.save(output_dir / "test_lgbm.npy", test_lgbm)

        metrics["models"]["lightgbm"] = {
            "folds": lgbm_metrics,
            "oof_macro_f1": macro_f1_from_probs(y, oof_lgbm),
        }

        oof_collection.append(oof_lgbm)
        test_collection.append(test_lgbm)

    if cfg["torch_models"].get("enabled", True):
        torch_cfg = cfg["torch_models"]

        oof_torch, test_torch, torch_metrics = train_torch_cv(
            x_train=x_train,
            y=y,
            x_test=x_test,
            n_splits=n_splits,
            seed=seed,
            epochs=int(torch_cfg.get("epochs", 25)),
            batch_size=int(torch_cfg.get("batch_size", 256)),
            lr=float(torch_cfg.get("lr", 1e-3)),
            weight_decay=float(torch_cfg.get("weight_decay", 1e-4)),
            patience=int(torch_cfg.get("patience", 5)),
            hidden_dim=int(torch_cfg.get("hidden_dim", 256)),
            dropout=float(torch_cfg.get("dropout", 0.15)),
            focal_gamma=float(torch_cfg.get("focal_gamma", 0.5)),
            label_smoothing=float(torch_cfg.get("label_smoothing", 0.05)),
        )
        np.save(output_dir / "oof_torch.npy", oof_torch)
        np.save(output_dir / "test_torch.npy", test_torch)

        metrics["models"]["torch"] = {
            "folds": torch_metrics,
            "oof_macro_f1": macro_f1_from_probs(y, oof_torch),
        }

        oof_collection.append(oof_torch)
        test_collection.append(test_torch)

    if len(oof_collection) == 0:
        raise RuntimeError("No model was enabled.")

    if len(oof_collection) == 1:
        oof_final = oof_collection[0]
        test_final = test_collection[0]
    else:
        weight_lgbm = float(cfg["ensemble"].get("weight_lgbm", 0.575))
        oof_final = blend_probs(oof_collection[0], oof_collection[1], weight_a=weight_lgbm)
        test_final = blend_probs(test_collection[0], test_collection[1], weight_a=weight_lgbm)

    np.save(output_dir / "oof_final.npy", oof_final)
    np.save(output_dir / "test_final.npy", test_final)

    final_oof_f1 = macro_f1_from_probs(y, oof_final)
    metrics["models"]["ensemble"] = {"oof_macro_f1": final_oof_f1}

    pred_labels = apply_postprocessing(
        probs=test_final,
        raw_test=test_df,
        config=cfg.get("postprocess", {}),
    )

    submission = sample_sub.copy()
    if target_col in submission.columns:
        submission[target_col] = pred_labels
    else:
        submission.iloc[:, -1] = pred_labels

    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    save_json(metrics, output_dir / "metrics.json")

    return {
        "metrics": metrics,
        "submission_path": str(submission_path),
    }
