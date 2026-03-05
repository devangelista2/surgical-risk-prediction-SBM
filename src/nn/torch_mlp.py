from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.logger import logger


def _to_numpy_array(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


class _TorchMLPBase(BaseEstimator):
    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        batch_size: int = 64,
        epochs: int = 120,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        validation_fraction: float = 0.2,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _build_network(self, input_dim: int, output_dim: int, nn) -> Any:
        activation_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }
        if self.activation not in activation_map:
            raise ValueError(
                f"Unsupported activation '{self.activation}'. "
                f"Choose from {sorted(activation_map.keys())}."
            )

        layers: list[Any] = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_map[self.activation]())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _resolve_device(self, torch):
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _build_eval_split(
        self,
        X_np: np.ndarray,
        y_idx: np.ndarray,
        eval_set: tuple[Any, Any] | None,
        n_classes: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if eval_set is not None:
            X_eval, y_eval_raw = eval_set
            X_eval_np = _to_numpy_array(X_eval)
            class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
            y_eval_idx = np.array(
                [class_to_idx.get(v, -1) for v in np.asarray(y_eval_raw)]
            )
            valid_mask = y_eval_idx >= 0
            if not np.any(valid_mask):
                return None, None
            return X_eval_np[valid_mask], y_eval_idx[valid_mask]

        if self.validation_fraction <= 0 or len(y_idx) < 8:
            return None, None

        stratify = y_idx if n_classes > 1 else None
        _, X_eval_np, _, y_eval_idx = train_test_split(
            X_np,
            y_idx,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=stratify,
        )
        return X_eval_np, y_eval_idx

    def _log_epoch_auc(
        self,
        epoch: int,
        n_classes: int,
        y_eval_idx: np.ndarray | None,
        probs: np.ndarray | None,
    ) -> None:
        if y_eval_idx is None or probs is None:
            return

        auc = np.nan
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_eval_idx, probs[:, 1])
            else:
                auc = roc_auc_score(y_eval_idx, probs, multi_class="ovr")
        except ValueError:
            auc = np.nan


class TorchMLPClassifier(_TorchMLPBase, ClassifierMixin):
    def fit(
        self, X: Any, y: Any, eval_set: tuple[Any, Any] | None = None
    ) -> TorchMLPClassifier:
        torch.manual_seed(self.random_state)

        X_np = _to_numpy_array(X)
        y_np = np.asarray(y)

        self.classes_, y_idx = np.unique(y_np, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("TorchMLPClassifier requires at least 2 classes.")

        self.n_features_in_ = X_np.shape[1]
        self.device_ = self._resolve_device(torch)
        X_eval_np, y_eval_idx = self._build_eval_split(X_np, y_idx, eval_set, n_classes)

        if n_classes == 2:
            self.model_ = self._build_network(self.n_features_in_, 1, nn)
            criterion = nn.BCEWithLogitsLoss()
            y_tensor = torch.tensor(y_idx.astype(np.float32)).view(-1, 1)
        else:
            self.model_ = self._build_network(self.n_features_in_, n_classes, nn)
            criterion = nn.CrossEntropyLoss()
            y_tensor = torch.tensor(y_idx.astype(np.int64))

        X_tensor = torch.tensor(X_np, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            if self.verbose:
                logger.info(
                    "[torch_mlp] epoch=%d train_loss=%.5f",
                    epoch + 1,
                    epoch_loss / len(loader),
                )

            if X_eval_np is not None and y_eval_idx is not None:
                probs = self.predict_proba(X_eval_np)
                self._log_epoch_auc(epoch + 1, n_classes, y_eval_idx, probs)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X_np = _to_numpy_array(X)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            if len(self.classes_) == 2:
                pos = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
                neg = 1.0 - pos
                return np.hstack([neg, pos])
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class TorchMLPRegressor(_TorchMLPBase, RegressorMixin):
    def fit(
        self, X: Any, y: Any, eval_set: tuple[Any, Any] | None = None
    ) -> TorchMLPRegressor:
        torch.manual_seed(self.random_state)

        X_np = _to_numpy_array(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.n_features_in_ = X_np.shape[1]
        self.device_ = self._resolve_device(torch)
        self.model_ = self._build_network(self.n_features_in_, 1, nn)

        criterion = nn.MSELoss()
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.to(self.device_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad()
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            if self.verbose:
                logger.info(
                    "[torch_mlp] epoch=%d train_loss=%.5f",
                    epoch + 1,
                    epoch_loss / len(loader),
                )

        return self

    def predict(self, X: Any) -> np.ndarray:
        X_np = _to_numpy_array(X)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X_tensor).cpu().numpy().reshape(-1)
        return pred
