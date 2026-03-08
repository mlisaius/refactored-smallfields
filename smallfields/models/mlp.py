from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight


class MLP(nn.Module):
    """3-layer MLP: (Linear → BatchNorm1d → ReLU → Dropout) × 3 → Linear output."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        num_classes: int,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        # Each hidden block: Linear → BatchNorm1d → ReLU → Dropout.
        # BatchNorm1d stabilises training by normalising activations per mini-batch;
        # Dropout regularises by randomly zeroing activations during training.
        # Architecture matches the source MLP scripts exactly (3 hidden layers).
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Output layer: raw logits (no softmax — CrossEntropyLoss includes log-softmax)
        self.output = nn.Linear(hidden_sizes[2], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


class MLPWrapper:
    """sklearn-compatible wrapper around a trained MLP.

    Handles the label-index reversal: internally the MLP uses 0-indexed labels;
    `predict()` adds `label_offset` back so predictions match the original label space.
    """

    def __init__(self, model: MLP, label_offset: int = 1, device=None):
        self.model = model
        # label_offset=1: MLP predicts 0-indexed classes; adding 1 restores 1-indexed labels
        self.label_offset = label_offset
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()  # disable Dropout and BatchNorm training behaviour
        X_tensor = torch.FloatTensor(X).to(self.device)
        predictions = []
        with torch.no_grad():  # disable gradient tracking for inference efficiency
            # Process in fixed-size batches to avoid OOM on large pixel arrays
            for i in range(0, len(X), 1024):
                batch = X_tensor[i : i + 1024]
                outputs = self.model(batch)
                # argmax over class logits → predicted 0-indexed class
                _, preds = torch.max(outputs, 1)
                # Shift back to original 1-indexed label space before returning
                predictions.append((preds + self.label_offset).cpu().numpy())
        return np.concatenate(predictions)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    input_size: int,
    hidden_sizes: list = None,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 1024,
    num_epochs: int = 50,
    patience: int = 5,
    use_class_weights: bool = True,
    label_shift: int = 1,
    device=None,
) -> MLP:
    """Train a 3-layer MLP with early stopping on validation loss.

    Parameters
    ----------
    label_shift : int
        Amount subtracted from y_train/y_val labels before creating tensors.
        E.g., label_shift=1 converts 1-indexed labels to 0-indexed for PyTorch.

    Returns
    -------
    MLP
        The best model (loaded from checkpoint at lowest validation loss).
    """
    if hidden_sizes is None:
        hidden_sizes = [512, 256, 128]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors and shift labels to 0-indexed for PyTorch CrossEntropyLoss
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train - label_shift)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val - label_shift)

    if use_class_weights:
        # compute_class_weight('balanced') computes weights inversely proportional to
        # class frequency, compensating for class imbalance in the training set.
        # weights are passed to CrossEntropyLoss so rare classes contribute more to the loss.
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        # Unweighted loss — used by austria_BTFM_mlp_smallfields.py
        criterion = nn.CrossEntropyLoss()

    # shuffle=True on training loader ensures different batch orderings each epoch
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    # shuffle=False on val loader gives a deterministic validation loss
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    model = MLP(input_size, hidden_sizes, num_classes, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # ReduceLROnPlateau: halve LR when val loss stops improving for 3 epochs.
    # Matches source script schedulers (factor=0.5, patience=3 are source defaults).
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model_state = None  # in-memory checkpoint (avoids writing to disk)

    for epoch in range(num_epochs):
        model.train()  # enable Dropout / BatchNorm training mode
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        # --- Validation pass ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_x), batch_y).item()
        # Average loss over all validation batches
        val_loss /= len(val_loader)

        # Let the scheduler observe this epoch's validation loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Deep-copy state dict so subsequent epochs don't corrupt the checkpoint
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            early_stop_counter = 0  # reset patience counter on improvement
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                # No improvement for `patience` consecutive epochs — stop early
                break

    # Restore the best weights found during training
    model.load_state_dict(best_model_state)
    return model
