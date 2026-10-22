"""
Graph Context Module for Scene Graph Generation.
================================================

Implements attention-based message passing between objects in a scene
to capture global context before relation prediction.

Based on approaches from:
- Graph R-CNN (Yang et al., ECCV 2018)
- GPS-Net (Lin et al., CVPR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GraphAttentionLayer(nn.Module):
    """
    Single graph attention layer for object-to-object message passing.

    Uses multi-head attention where each object attends to all other
    objects in the scene to gather contextual information.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_edge_features: Whether to use edge (spatial) features in attention
        edge_dim: Dimension of edge features if used
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        edge_dim: int = 128
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_edge_features = use_edge_features

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

        # Edge feature projection for attention bias
        if use_edge_features:
            self.edge_proj = nn.Sequential(
                nn.Linear(edge_dim, num_heads),
                nn.Tanh()
            )

        # Output projection
        self.out_proj = nn.Linear(output_dim, output_dim)

        # Layer norm and dropout
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Scale factor for attention
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-head self-attention.

        Args:
            x: Node features [batch, num_nodes, input_dim] or [num_nodes, input_dim]
            edge_features: Optional edge features [batch, num_nodes, num_nodes, edge_dim]
                          or [num_nodes, num_nodes, edge_dim]
            attention_mask: Optional mask [batch, num_nodes, num_nodes]

        Returns:
            Tuple of (output_features, attention_weights)
        """
        # Handle 2D input (single graph)
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True
            if edge_features is not None and edge_features.dim() == 3:
                edge_features = edge_features.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0)

        batch_size, num_nodes, _ = x.shape

        # Compute Q, K, V
        Q = self.query(x)  # [B, N, D]
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [B, H, N, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [B, H, N, N]

        # Add edge feature bias if available
        if self.use_edge_features and edge_features is not None:
            edge_bias = self.edge_proj(edge_features)  # [B, N, N, H]
            edge_bias = edge_bias.permute(0, 3, 1, 2)   # [B, H, N, N]
            attn_scores = attn_scores + edge_bias

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.unsqueeze(1), float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [B, H, N, head_dim]

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.output_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection and layer norm
        out = self.norm(x + out) if x.shape[-1] == self.output_dim else self.norm(out)

        if squeeze_batch:
            out = out.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return out, attn_weights


class GraphContextEncoder(nn.Module):
    """
    Multi-layer graph context encoder.

    Stacks multiple graph attention layers for iterative message passing
    to capture multi-hop context information.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_layers: Number of attention layers
        num_heads: Number of attention heads per layer
        dropout: Dropout probability
        use_edge_features: Whether to incorporate spatial edge features
        edge_dim: Dimension of edge features
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        edge_dim: int = 128
    ):
        super().__init__()
        self.num_layers = num_layers

        # Input projection if needed
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # Stack of graph attention layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = hidden_dim
            layer_output_dim = hidden_dim if i < num_layers - 1 else output_dim

            self.layers.append(
                GraphAttentionLayer(
                    input_dim=layer_input_dim,
                    output_dim=layer_output_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_edge_features=use_edge_features,
                    edge_dim=edge_dim
                )
            )

        # Final layer norm
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through all layers.

        Args:
            x: Node features [batch, num_nodes, input_dim] or [num_nodes, input_dim]
            edge_features: Optional edge features
            attention_mask: Optional attention mask

        Returns:
            Tuple of (output_features, list_of_attention_weights)
        """
        # Project input
        x = self.input_proj(x)

        all_attn_weights = []

        # Pass through layers
        for layer in self.layers:
            x, attn_weights = layer(x, edge_features, attention_mask)
            all_attn_weights.append(attn_weights)

        # Final normalization
        x = self.final_norm(x)

        return x, all_attn_weights


class EdgeContextEncoder(nn.Module):
    """
    Encoder for relation/edge-level context.

    After objects have been enriched with context, this module
    creates context-aware edge representations for each pair.

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of spatial edge features
        output_dim: Output edge feature dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        node_dim: int = 256,
        edge_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Combine subject, object, and edge features
        combined_dim = node_dim * 2 + edge_dim

        # Attention over all edges for context
        self.edge_attention = nn.MultiheadAttention(
            embed_dim=combined_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        subj_features: torch.Tensor,
        obj_features: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute context-aware edge representations.

        Args:
            subj_features: Subject node features [N, node_dim]
            obj_features: Object node features [N, node_dim]
            edge_features: Spatial edge features [N, edge_dim]

        Returns:
            Context-aware edge features [N, output_dim]
        """
        # Combine features for each edge
        combined = torch.cat([subj_features, obj_features, edge_features], dim=-1)
        # [N, node_dim * 2 + edge_dim]

        # Add batch dimension for attention
        combined = combined.unsqueeze(0)  # [1, N, D]

        # Self-attention over all edges
        attn_out, _ = self.edge_attention(combined, combined, combined)

        # Residual and project
        out = combined + attn_out
        out = out.squeeze(0)  # [N, D]
        out = self.output_proj(out)
        out = self.norm(out)

        return out


class GraphContextModule(nn.Module):
    """
    Complete graph context module for scene graph generation.

    Combines:
    1. Object-level message passing (GraphContextEncoder)
    2. Edge-level context aggregation (EdgeContextEncoder)

    Args:
        node_dim: Object feature dimension
        edge_dim: Spatial feature dimension
        hidden_dim: Hidden dimension
        num_object_layers: Number of object-level attention layers
        num_edge_layers: Number of edge-level attention layers
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        node_dim: int = 256,
        edge_dim: int = 128,
        hidden_dim: int = 256,
        num_object_layers: int = 2,
        num_edge_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Object-level context encoder
        self.object_encoder = GraphContextEncoder(
            input_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_object_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_edge_features=True,
            edge_dim=edge_dim
        )

        # Edge-level context encoder
        self.edge_encoder = EdgeContextEncoder(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        pairs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process objects and edges with context.

        Args:
            node_features: Object features [num_objects, node_dim]
            edge_features: Full edge feature matrix [num_objects, num_objects, edge_dim]
            pairs: Relation pairs [num_pairs, 2] with (subj_idx, obj_idx)

        Returns:
            Tuple of:
                - Context-enriched node features [num_objects, hidden_dim]
                - Context-enriched edge features [num_pairs, hidden_dim]
        """
        # Object-level message passing
        context_nodes, _ = self.object_encoder(node_features, edge_features)

        # Extract subject and object features for each pair
        subj_idx = pairs[:, 0]
        obj_idx = pairs[:, 1]

        subj_features = context_nodes[subj_idx]
        obj_features = context_nodes[obj_idx]

        # Get spatial features for each pair
        pair_edge_features = edge_features[subj_idx, obj_idx]

        # Edge-level context
        context_edges = self.edge_encoder(subj_features, obj_features, pair_edge_features)

        return context_nodes, context_edges


def compute_edge_features(
    boxes: torch.Tensor,
    spatial_encoder: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Compute pairwise edge features for all object pairs.

    Args:
        boxes: Object boxes [N, 4] in (x1, y1, x2, y2) format
        spatial_encoder: Optional encoder to project spatial features

    Returns:
        Edge features [N, N, edge_dim]
    """
    N = boxes.size(0)
    device = boxes.device

    # Compute box centers and sizes
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    # Pairwise relative positions
    # [N, N] matrices
    dx = cx.unsqueeze(1) - cx.unsqueeze(0)  # subject - object
    dy = cy.unsqueeze(1) - cy.unsqueeze(0)
    dw = torch.log((w.unsqueeze(1) / w.unsqueeze(0)).clamp(min=1e-6))
    dh = torch.log((h.unsqueeze(1) / h.unsqueeze(0)).clamp(min=1e-6))

    # Normalize by subject box size
    dx = dx / w.unsqueeze(1).clamp(min=1e-6)
    dy = dy / h.unsqueeze(1).clamp(min=1e-6)

    # Stack features [N, N, 4]
    edge_features = torch.stack([dx, dy, dw, dh], dim=-1)

    # Apply encoder if provided
    if spatial_encoder is not None:
        orig_shape = edge_features.shape
        edge_features = edge_features.view(-1, 4)
        edge_features = spatial_encoder(edge_features)
        edge_features = edge_features.view(N, N, -1)

    return edge_features
