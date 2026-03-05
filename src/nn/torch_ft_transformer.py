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


class FTTransformerNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        out_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.feature_weight = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.xavier_uniform_(self.feature_weight)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Embedding(n_features + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):
        bsz, n_feats = x.shape
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(
            0
        ) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(bsz, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)

        pos_ids = (
            torch.arange(n_feats + 1, device=x.device).unsqueeze(0).expand(bsz, -1)
        )
        seq = seq + self.position(pos_ids)
        seq = self.encoder(seq)
        cls_out = seq[:, 0, :]
        return self.head(cls_out)


class _TorchFTBase(BaseEstimator):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        batch_size: int = 128,
        epochs: int = 120,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        validation_fraction: float = 0.2,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
        pos_weight: float | None = None,
        class_weights: tuple[float, ...] | None = None,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.pos_weight = pos_weight
        self.class_weights = class_weights

    def _resolve_device(self) -> str:
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

        logger.info(
            "[torch_ft_transformer] epoch=%d val_auc_roc=%s",
            epoch,
            f"{auc:.5f}" if np.isfinite(auc) else "nan",
        )


class TorchFTTransformerClassifier(_TorchFTBase, ClassifierMixin):
    def fit(
        self, X: Any, y: Any, eval_set: tuple[Any, Any] | None = None
    ) -> TorchFTTransformerClassifier:
        torch.manual_seed(self.random_state)

        X_np = _to_numpy_array(X)
        y_np = np.asarray(y)

        self.classes_, y_idx = np.unique(y_np, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                "TorchFTTransformerClassifier requires at least 2 classes."
            )

        self.n_features_in_ = X_np.shape[1]
        self.device_ = self._resolve_device()
        X_eval_np, y_eval_idx = self._build_eval_split(X_np, y_idx, eval_set, n_classes)

        out_dim = 1 if n_classes == 2 else n_classes
        self.model_ = FTTransformerNet(
            n_features=self.n_features_in_,
            out_dim=out_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        ).to(self.device_)

        if n_classes == 2:
            if self.pos_weight is not None:
                pos_w = torch.tensor(
                    [float(self.pos_weight)], dtype=torch.float32, device=self.device_
                )
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            else:
                criterion = nn.BCEWithLogitsLoss()
            y_tensor = torch.tensor(y_idx.astype(np.float32)).view(-1, 1)
        else:
            weight_tensor = None
            if self.class_weights is not None and len(self.class_weights) == n_classes:
                weight_tensor = torch.tensor(
                    [float(w) for w in self.class_weights],
                    dtype=torch.float32,
                    device=self.device_,
                )
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            y_tensor = torch.tensor(y_idx.astype(np.int64))

        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
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
                    "[torch_ft_transformer] epoch=%d train_loss=%.5f",
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


class TorchFTTransformerRegressor(_TorchFTBase, RegressorMixin):
    def fit(
        self, X: Any, y: Any, eval_set: tuple[Any, Any] | None = None
    ) -> TorchFTTransformerRegressor:
        torch.manual_seed(self.random_state)

        X_np = _to_numpy_array(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.n_features_in_ = X_np.shape[1]
        self.device_ = self._resolve_device()

        self.model_ = FTTransformerNet(
            n_features=self.n_features_in_,
            out_dim=1,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        ).to(self.device_)

        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
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
                    "[torch_ft_transformer] epoch=%d train_loss=%.5f",
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
