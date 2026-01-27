"""Neural Network Expert for MoE architecture.

This module implements a PyTorch-based TabularMLP with residual connections
for deep pattern learning on tabular market data.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Neural expert using device: {DEVICE}")


class ResidualBlock(nn.Module):
    """Residual block with skip connection for deep learning on tabular data."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip connection (project if dimensions differ)
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        # Residual connection
        out = out + identity
        return out


class TabularMLP(nn.Module):
    """ResNet-style MLP for tabular data with skip connections.

    Architecture:
        Input → BatchNorm → ResBlock1 → ResBlock2 → ResBlock3 → Output Head

    Anti-overfitting features:
        - Batch normalization after every layer
        - Dropout in each residual block
        - Skip connections for gradient flow
        - GELU activation (smoother than ReLU)
    """

    def __init__(
        self,
        input_dim: int = 52,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(0.2)

        # Build residual blocks
        self.res_blocks = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.res_blocks.append(
                ResidualBlock(prev_dim, hidden_dim, dropout)
            )
            prev_dim = hidden_dim

        # Output head
        self.output_fc1 = nn.Linear(hidden_dims[-1], 32)
        self.output_bn = nn.BatchNorm1d(32)
        self.output_fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalization
        x = self.input_bn(x)
        x = self.input_dropout(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Output head
        x = self.output_fc1(x)
        x = self.output_bn(x)
        x = F.gelu(x)
        x = self.output_fc2(x)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing for regularization."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_classes = logits.size(-1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute loss
        loss = (-smooth_targets * log_probs).sum(dim=-1)

        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights

        return loss.mean()


class NeuralExpertTrainer:
    """Trainer for neural network experts with anti-overfitting strategies."""

    def __init__(
        self,
        input_dim: int = 52,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 200,
        early_stopping_patience: int = 15,
        label_smoothing: float = 0.1,
        device: torch.device = DEVICE,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.label_smoothing = label_smoothing
        self.device = device

        self.model: Optional[TabularMLP] = None
        self.scaler = None  # Will store StandardScaler from sklearn

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[TabularMLP, dict]:
        """Train the neural network with early stopping.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - binary 0/1
            X_val: Validation features
            y_val: Validation labels
            sample_weights: Optional sample weights for training

        Returns:
            Tuple of (trained model, training metrics dict)
        """
        # Determine input dimension from data
        actual_input_dim = X_train.shape[1]

        # Initialize model
        self.model = TabularMLP(
            input_dim=actual_input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)

        if sample_weights is not None:
            weights_t = torch.FloatTensor(sample_weights).to(self.device)
        else:
            weights_t = None

        # Create data loaders
        if weights_t is not None:
            train_dataset = TensorDataset(X_train_t, y_train_t, weights_t)
        else:
            train_dataset = TensorDataset(X_train_t, y_train_t)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > self.batch_size,
        )

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=False,
        )

        # Loss function with label smoothing
        criterion = LabelSmoothingCrossEntropy(smoothing=self.label_smoothing)

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        training_history = []

        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                if weights_t is not None:
                    X_batch, y_batch, w_batch = batch
                else:
                    X_batch, y_batch = batch
                    w_batch = None

                # Data augmentation: add Gaussian noise
                if self.training:
                    X_batch = X_batch + torch.randn_like(X_batch) * 0.01

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch, w_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item() * len(X_batch)
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == y_batch).sum().item()
                train_total += len(y_batch)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss = F.cross_entropy(val_logits, y_val_t).item()
                val_predictions = torch.argmax(val_logits, dim=1)
                val_acc = (val_predictions == y_val_t).float().mean().item()

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Record history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr'],
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Log progress every 20 epochs
            if epoch % 20 == 0:
                logger.debug(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
                )

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        metrics = {
            'final_train_loss': training_history[-1]['train_loss'],
            'final_train_acc': training_history[-1]['train_acc'],
            'best_val_loss': best_val_loss,
            'final_val_acc': training_history[-1]['val_acc'],
            'epochs_trained': len(training_history),
            'early_stopped': patience_counter >= self.early_stopping_patience,
        }

        logger.info(
            f"Training complete: {metrics['epochs_trained']} epochs, "
            f"val_loss={best_val_loss:.4f}, val_acc={metrics['final_val_acc']:.3f}"
        )

        return self.model, metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for input features.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, 2) for binary classification
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            probs = self.model.predict_proba(X_t)
            return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions for input features."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, path: str) -> None:
        """Save model state to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'hidden_dims': self.model.hidden_dims,
            'dropout': self.model.dropout_rate,
            'config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'early_stopping_patience': self.early_stopping_patience,
                'label_smoothing': self.label_smoothing,
            }
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: torch.device = DEVICE) -> "NeuralExpertTrainer":
        """Load model from disk."""
        state = torch.load(path, map_location=device)

        trainer = cls(
            input_dim=state['input_dim'],
            hidden_dims=state['hidden_dims'],
            dropout=state['dropout'],
            device=device,
            **state.get('config', {})
        )

        trainer.model = TabularMLP(
            input_dim=state['input_dim'],
            hidden_dims=state['hidden_dims'],
            dropout=state['dropout'],
        ).to(device)
        trainer.model.load_state_dict(state['model_state_dict'])

        return trainer


def get_neural_trainer(
    input_dim: int = 52,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 128,
    max_epochs: int = 200,
    early_stopping_patience: int = 15,
) -> NeuralExpertTrainer:
    """Factory function to create a neural expert trainer."""
    return NeuralExpertTrainer(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
    )
