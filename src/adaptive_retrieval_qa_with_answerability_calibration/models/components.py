"""Custom neural network components for Adaptive Retrieval QA.

This module provides reusable building blocks for the answerability calibration system,
including attention mechanisms, feature extractors, and fusion layers.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for aggregating sequence representations.

    This component uses multi-head attention to compute a weighted sum over
    sequence elements, providing a context-aware pooling mechanism that's
    more powerful than simple max or mean pooling.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ) -> None:
        """Initialize multi-head attention pooling.

        Args:
            hidden_size: Hidden dimension size.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Query vector for pooling
        self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim))

        # Key and value projections
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply multi-head attention pooling.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size].
            attention_mask: Optional attention mask [batch_size, seq_len].

        Returns:
            Pooled representation [batch_size, hidden_size].
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project to key and value
        keys = self.key_proj(hidden_states)  # [batch, seq_len, hidden]
        values = self.value_proj(hidden_states)  # [batch, seq_len, hidden]

        # Reshape for multi-head attention
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # keys/values: [batch, num_heads, seq_len, head_dim]

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1).unsqueeze(2)  # [batch, num_heads, 1, head_dim]

        # Compute attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [batch, num_heads, 1, seq_len]

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for heads: [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, 1, seq_len]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        pooled = torch.matmul(attn_weights, values)  # [batch, num_heads, 1, head_dim]

        # Concatenate heads
        pooled = pooled.transpose(1, 2).contiguous().view(batch_size, self.hidden_size)

        # Apply layer normalization
        pooled = self.layer_norm(pooled)

        return pooled


class ConfidenceFeatureExtractor(nn.Module):
    """Extracts confidence-related features from QA model outputs.

    This component computes various statistical and distributional features
    from QA model logits and hidden states to help calibrate confidence scores.
    """

    def __init__(self, hidden_size: int, output_dim: int = 10) -> None:
        """Initialize confidence feature extractor.

        Args:
            hidden_size: Hidden dimension of input.
            output_dim: Output feature dimension.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # Feature projection layers
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size + 8, output_dim * 2),  # +8 for computed features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor
    ) -> torch.Tensor:
        """Extract confidence features.

        Args:
            hidden_states: Hidden states [batch_size, hidden_size].
            start_logits: Start position logits [batch_size, seq_len].
            end_logits: End position logits [batch_size, seq_len].

        Returns:
            Confidence features [batch_size, output_dim].
        """
        batch_size = hidden_states.size(0)

        # Convert logits to probabilities
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)

        # Statistical features from logits
        start_max = torch.max(start_probs, dim=-1)[0]
        start_entropy = -torch.sum(start_probs * torch.log(start_probs + 1e-9), dim=-1)
        end_max = torch.max(end_probs, dim=-1)[0]
        end_entropy = -torch.sum(end_probs * torch.log(end_probs + 1e-9), dim=-1)

        # Top-k confidence gap (difference between best and second-best)
        start_sorted, _ = torch.sort(start_probs, descending=True, dim=-1)
        end_sorted, _ = torch.sort(end_probs, descending=True, dim=-1)
        start_gap = start_sorted[:, 0] - start_sorted[:, 1]
        end_gap = end_sorted[:, 0] - end_sorted[:, 1]

        # Spread of probability mass
        start_std = torch.std(start_probs, dim=-1)
        end_std = torch.std(end_probs, dim=-1)

        # Stack statistical features
        stat_features = torch.stack([
            start_max, start_entropy, start_gap, start_std,
            end_max, end_entropy, end_gap, end_std
        ], dim=-1)  # [batch_size, 8]

        # Concatenate with hidden states
        combined_features = torch.cat([hidden_states, stat_features], dim=-1)

        # Project to output dimension
        features = self.feature_projection(combined_features)

        return features


class GatedFusionLayer(nn.Module):
    """Gated fusion layer for combining multiple feature sources.

    Uses learned gates to dynamically weight the contribution of different
    feature sources (e.g., retrieval scores, QA confidence, semantic features).
    """

    def __init__(self, input_dims: list, output_dim: int, dropout: float = 0.1) -> None:
        """Initialize gated fusion layer.

        Args:
            input_dims: List of input feature dimensions.
            output_dim: Output dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.num_sources = len(input_dims)
        self.input_dims = input_dims
        self.output_dim = output_dim

        # Individual projection layers for each source
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

        # Gating network
        total_dim = sum(input_dims)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, self.num_sources),
            nn.Softmax(dim=-1)
        )

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *inputs) -> torch.Tensor:
        """Fuse multiple input features.

        Args:
            *inputs: Variable number of input tensors, each [batch_size, input_dim].

        Returns:
            Fused features [batch_size, output_dim].
        """
        assert len(inputs) == self.num_sources, f"Expected {self.num_sources} inputs, got {len(inputs)}"

        batch_size = inputs[0].size(0)

        # Project each input to output dimension
        projected = [proj(inp) for proj, inp in zip(self.projections, inputs)]
        projected = torch.stack(projected, dim=1)  # [batch, num_sources, output_dim]

        # Compute gates from concatenated inputs
        concatenated = torch.cat(inputs, dim=-1)  # [batch, total_dim]
        gates = self.gate(concatenated)  # [batch, num_sources]
        gates = gates.unsqueeze(-1)  # [batch, num_sources, 1]

        # Apply gated fusion
        fused = torch.sum(projected * gates, dim=1)  # [batch, output_dim]

        # Normalize and dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused


class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration.

    Learns a temperature parameter to rescale logits, improving calibration
    of predicted probabilities. Based on "On Calibration of Modern Neural Networks".
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        """Initialize temperature scaling.

        Args:
            init_temperature: Initial temperature value.
        """
        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(init_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling.

        Args:
            logits: Input logits [batch_size, ...].

        Returns:
            Scaled logits [batch_size, ...].
        """
        # Clamp temperature to avoid numerical issues
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / temp

    def get_temperature(self) -> float:
        """Get current temperature value.

        Returns:
            Temperature value.
        """
        return self.temperature.item()


class ResidualMLP(nn.Module):
    """Residual MLP block for deep feature transformation.

    Implements a multi-layer perceptron with residual connections,
    batch normalization, and dropout for stable deep learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ) -> None:
        """Initialize residual MLP.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (defaults to input_dim if None).
            num_layers: Number of hidden layers.
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        output_dim = output_dim or input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection if dimensions differ
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Hidden layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_block = []
            layer_block.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layer_block.append(nn.BatchNorm1d(hidden_dim))
            layer_block.append(nn.ReLU())
            layer_block.append(nn.Dropout(dropout))
            self.layers.append(nn.Sequential(*layer_block))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

        # Skip connection projection if needed
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor [batch_size, input_dim].

        Returns:
            Output tensor [batch_size, output_dim].
        """
        identity = x

        # Input projection
        out = self.input_proj(x)

        # Apply hidden layers with residual connections
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual  # Residual connection

        # Output projection
        out = self.output_proj(out)

        # Skip connection from input
        out = out + self.skip_proj(identity)

        return out


# Export all components
__all__ = [
    'MultiHeadAttentionPooling',
    'ConfidenceFeatureExtractor',
    'GatedFusionLayer',
    'TemperatureScaling',
    'ResidualMLP'
]
