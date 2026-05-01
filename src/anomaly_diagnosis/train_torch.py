from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from anomaly_diagnosis.torch_models import FocalLoss, SingleTokenTransformer, TabularMLP


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_proba(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probs = []
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.vstack(probs)


def train_torch_cv(
    x_train: pd.DataFrame,
    y: pd.Series,
    x_test: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    hidden_dim: int = 256,
    dropout: float = 0.15,
    focal_gamma: float = 0.5,
    label_smoothing: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = np.sort(pd.Series(y).unique())
    num_classes = len(classes)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros((len(x_train), num_classes), dtype=float)
    test_probs = np.zeros((len(x_test), num_classes), dtype=float)
    metrics = []

    x_np_raw = x_train.values.astype(np.float32)
    x_test_raw = x_test.values.astype(np.float32)
    y_np = y.values.astype(np.int64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_np_raw, y_np), start=1):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_np_raw[tr_idx])
        x_va = scaler.transform(x_np_raw[va_idx])
        x_te = scaler.transform(x_test_raw)

        y_tr = y_np[tr_idx]
        y_va = y_np[va_idx]

        fold_test_probs = []
        fold_val_probs = []

        for model_name in ["mlp", "transformer"]:
            torch.manual_seed(seed + fold)

            if model_name == "mlp":
                model = TabularMLP(
                    input_dim=x_tr.shape[1],
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
            else:
                model = SingleTokenTransformer(
                    input_dim=x_tr.shape[1],
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )

            model = model.to(device)
            criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2,
            )

            train_loader = _make_loader(x_tr, y_tr, batch_size=batch_size, shuffle=True)

            best_score = -1.0
            best_state = None
            bad_epochs = 0

            for _epoch in range(1, epochs + 1):
                model.train()

                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                va_probs = _predict_proba(model, x_va, device, batch_size)
                va_pred = va_probs.argmax(axis=1)
                score = f1_score(y_va, va_pred, average="macro")
                scheduler.step(score)

                if score > best_score:
                    best_score = score
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if bad_epochs >= patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            fold_val_probs.append(_predict_proba(model, x_va, device, batch_size))
            fold_test_probs.append(_predict_proba(model, x_te, device, batch_size))

        va_probs_avg = np.mean(fold_val_probs, axis=0)
        te_probs_avg = np.mean(fold_test_probs, axis=0)

        oof[va_idx] = va_probs_avg
        test_probs += te_probs_avg / n_splits

        fold_pred = va_probs_avg.argmax(axis=1)
        macro_f1 = f1_score(y_va, fold_pred, average="macro")
        metrics.append({"fold": fold, "macro_f1": float(macro_f1)})

    return oof, test_probs, metrics
