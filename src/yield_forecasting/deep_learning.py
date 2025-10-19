"""Optional PyTorch multi-task regressor.

The main pipeline uses scikit-learn for fast, auditable tabular model selection. This module
demonstrates the deep-learning alternative without making a large runtime a core dependency.
"""

from __future__ import annotations

from typing import Any


def build_mlp(input_dim: int, output_dim: int = 2) -> Any:
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install with: pip install -e '.[deep-learning]'") from exc
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, output_dim),
    )


def train_mlp(x_train: Any, y_train: Any, epochs: int = 100, seed: int = 42) -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install with: pip install -e '.[deep-learning]'") from exc
    torch.manual_seed(seed)
    x = torch.as_tensor(x_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.float32)
    model = build_mlp(x.shape[1], y.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
    model.eval()
    return model
