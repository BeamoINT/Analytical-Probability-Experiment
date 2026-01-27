"""Neural Gating Network with Attention for MoE.

Implements an attention-based gating mechanism that learns to route
samples to the most appropriate experts based on input features.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)

# Device selection
if TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for gating features."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
               For single vector input, seq_len=1

        Returns:
            Attended output (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)

        return out


class AttentionGatingNetwork(nn.Module):
    """Attention-based gating network for expert selection.

    Architecture:
        Input(gating_features) → Linear(64) + LayerNorm + GELU
            ↓
        Multi-Head Self-Attention (4 heads)
            ↓
        Linear(32) + LayerNorm + GELU → Linear(num_experts) → Softmax
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Self-attention
        self.attention = MultiHeadSelfAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc1_norm = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, num_experts)

        self.dropout = nn.Dropout(dropout)

        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gating weights for experts.

        Args:
            x: Gating features (batch_size, input_dim)

        Returns:
            Expert weights (batch_size, num_experts) summing to 1
        """
        # Input projection
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.gelu(h)
        h = self.dropout(h)

        # Add sequence dimension for attention
        h = h.unsqueeze(1)  # (batch, 1, hidden)

        # Self-attention with residual
        attn_out = self.attention(h)
        h = self.attention_norm(h + attn_out)

        # Remove sequence dimension
        h = h.squeeze(1)  # (batch, hidden)

        # Output layers
        h = self.fc1(h)
        h = self.fc1_norm(h)
        h = F.gelu(h)
        h = self.dropout(h)

        # Final projection to expert weights
        logits = self.fc2(h)

        # Temperature-scaled softmax
        weights = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)

        return weights

    def get_top_k_experts(
        self,
        x: torch.Tensor,
        k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k experts and their weights.

        Args:
            x: Gating features (batch_size, input_dim)
            k: Number of top experts to return

        Returns:
            Tuple of (expert_indices, expert_weights) both (batch_size, k)
        """
        weights = self.forward(x)
        top_weights, top_indices = torch.topk(weights, k=min(k, self.num_experts), dim=-1)

        # Renormalize weights
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        return top_indices, top_weights


class NeuralGatingTrainer:
    """Trainer for the attention-based gating network."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        device: torch.device = DEVICE,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural gating")

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device

        self.model: Optional[AttentionGatingNetwork] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        expert_performance: Optional[np.ndarray] = None,
    ) -> Tuple["AttentionGatingNetwork", Dict]:
        """Train the gating network.

        Args:
            X_train: Training gating features (n_samples, n_gating_features)
            y_train: Training labels - best expert index (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            expert_performance: Optional soft labels (n_samples, n_experts)
                showing performance of each expert on each sample

        Returns:
            Tuple of (trained model, training metrics)
        """
        # Initialize model
        self.model = AttentionGatingNetwork(
            input_dim=self.input_dim,
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)

        # Use soft labels if provided
        if expert_performance is not None:
            soft_labels_t = torch.FloatTensor(expert_performance).to(self.device)
            train_dataset = TensorDataset(X_train_t, y_train_t, soft_labels_t)
            use_soft_labels = True
        else:
            train_dataset = TensorDataset(X_train_t, y_train_t)
            use_soft_labels = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > self.batch_size,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        history = []

        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                if use_soft_labels:
                    X_batch, y_batch, soft_batch = batch
                else:
                    X_batch, y_batch = batch
                    soft_batch = None

                optimizer.zero_grad()

                # Forward pass
                weights = self.model(X_batch)

                # Loss: combine hard and soft labels
                if soft_batch is not None:
                    # KL divergence for soft labels
                    soft_loss = F.kl_div(
                        torch.log(weights + 1e-8),
                        soft_batch,
                        reduction='batchmean'
                    )
                    # Cross entropy for hard labels
                    hard_loss = F.cross_entropy(torch.log(weights + 1e-8), y_batch)
                    loss = 0.7 * soft_loss + 0.3 * hard_loss
                else:
                    loss = F.cross_entropy(torch.log(weights + 1e-8), y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(X_batch)
                predictions = weights.argmax(dim=1)
                train_correct += (predictions == y_batch).sum().item()
                train_total += len(y_batch)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_weights = self.model(X_val_t)
                val_loss = F.cross_entropy(
                    torch.log(val_weights + 1e-8), y_val_t
                ).item()
                val_predictions = val_weights.argmax(dim=1)
                val_acc = (val_predictions == y_val_t).float().mean().item()

            scheduler.step(val_loss)

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Gating network early stopped at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                logger.debug(
                    f"Gating epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
                )

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        metrics = {
            'final_train_acc': history[-1]['train_acc'],
            'best_val_loss': best_val_loss,
            'final_val_acc': history[-1]['val_acc'],
            'epochs_trained': len(history),
        }

        logger.info(
            f"Gating training complete: {metrics['epochs_trained']} epochs, "
            f"val_acc={metrics['final_val_acc']:.3f}"
        )

        return self.model, metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get expert weights for samples.

        Args:
            X: Gating features (n_samples, n_gating_features)

        Returns:
            Expert weights (n_samples, n_experts)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            weights = self.model(X_t)
            return weights.cpu().numpy()

    def predict_top_k(
        self,
        X: np.ndarray,
        k: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k experts and weights.

        Args:
            X: Gating features
            k: Number of top experts

        Returns:
            Tuple of (expert_indices, expert_weights)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            indices, weights = self.model.get_top_k_experts(X_t, k)
            return indices.cpu().numpy(), weights.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'num_experts': self.num_experts,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
            }
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: torch.device = DEVICE) -> "NeuralGatingTrainer":
        """Load model from disk."""
        state = torch.load(path, map_location=device)

        trainer = cls(
            input_dim=state['input_dim'],
            num_experts=state['num_experts'],
            hidden_dim=state['hidden_dim'],
            num_heads=state['num_heads'],
            dropout=state['dropout'],
            device=device,
            **state.get('config', {})
        )

        trainer.model = AttentionGatingNetwork(
            input_dim=state['input_dim'],
            num_experts=state['num_experts'],
            hidden_dim=state['hidden_dim'],
            num_heads=state['num_heads'],
            dropout=state['dropout'],
        ).to(device)
        trainer.model.load_state_dict(state['model_state_dict'])

        return trainer


class HybridGating:
    """Hybrid gating that combines neural attention with gradient boosting.

    Uses neural gating for complex patterns and falls back to
    gradient boosting for interpretability and stability.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        neural_weight: float = 0.6,
        use_neural: bool = True,
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.neural_weight = neural_weight
        self.use_neural = use_neural and TORCH_AVAILABLE

        self.neural_trainer: Optional[NeuralGatingTrainer] = None
        self.gb_model = None  # Gradient boosting fallback

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        expert_performance: Optional[np.ndarray] = None,
    ) -> Dict:
        """Train both neural and GB gating models."""
        metrics = {}

        # Train neural gating
        if self.use_neural:
            self.neural_trainer = NeuralGatingTrainer(
                input_dim=self.input_dim,
                num_experts=self.num_experts,
            )
            _, neural_metrics = self.neural_trainer.train(
                X_train, y_train, X_val, y_val, expert_performance
            )
            metrics['neural'] = neural_metrics

        # Train gradient boosting fallback
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            self.gb_model = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_depth=6,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
            )
            self.gb_model.fit(X_train, y_train)
            gb_acc = self.gb_model.score(X_val, y_val)
            metrics['gb'] = {'val_acc': gb_acc}
        except Exception as e:
            logger.warning(f"GB gating training failed: {e}")
            self.gb_model = None

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get expert weights combining neural and GB predictions."""
        weights = np.zeros((len(X), self.num_experts))

        # Neural predictions
        if self.use_neural and self.neural_trainer is not None:
            neural_weights = self.neural_trainer.predict(X)
            weights += self.neural_weight * neural_weights

        # GB predictions
        if self.gb_model is not None:
            gb_proba = self.gb_model.predict_proba(X)
            # Map class probabilities to expert weights
            for i, cls in enumerate(self.gb_model.classes_):
                if cls < self.num_experts:
                    weights[:, cls] += (1 - self.neural_weight) * gb_proba[:, i]

        # Normalize
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        return weights
