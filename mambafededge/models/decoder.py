"""
Mamba-2 Decoder for on-device battery health inference.

Works with summary vectors from the encoder to produce predictions.
Accepts prompts from the Long Sequence Generator for fallback queries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from mambafededge.models.mamba2 import Mamba2Layer


class PromptProcessor(nn.Module):
    """Processes prompts from the Long Sequence Generator.

    Takes extreme stress points and encoded data summaries
    to create a structured prompt for the decoder.
    """

    def __init__(self, d_model: int, max_prompt_len: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_prompt_len = max_prompt_len

        self.prompt_encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        # Learnable prompt tokens for structuring the query
        self.prompt_prefix = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)

    def forward(
        self,
        summary_vectors: torch.Tensor,
        stress_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            summary_vectors: (B, n_summary, d_model) from encoder
            stress_context: (B, n_stress, d_model) extreme stress embeddings
        Returns:
            prompt: (B, prompt_len, d_model)
        """
        B = summary_vectors.shape[0]
        prefix = self.prompt_prefix.expand(B, -1, -1)

        parts = [prefix, summary_vectors]
        if stress_context is not None:
            parts.append(stress_context)

        prompt = torch.cat(parts, dim=1)
        prompt = self.prompt_encoder(prompt)
        return prompt[:, :self.max_prompt_len]


class Mamba2Decoder(nn.Module):
    """Mamba-2 Decoder for on-device inference.

    Designed for fast inference on edge devices:
    - Accepts summary vectors as context
    - Processes prompts for fallback queries
    - Produces multi-task outputs
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        n_outputs: int = 4,  # SOH, RUL, degradation params, thermal
    ):
        super().__init__()
        self.d_model = d_model
        self.n_outputs = n_outputs

        self.prompt_processor = PromptProcessor(d_model)

        # Cross-attention to attend to encoder summary
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Mamba-2 decoder layers
        self.layers = nn.ModuleList([
            Mamba2Layer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output heads
        self.output_heads = nn.ModuleDict({
            "soh": nn.Linear(d_model, 1),         # State of Health [0, 1]
            "rul": nn.Linear(d_model, 1),          # Remaining Useful Life
            "degradation": nn.Linear(d_model, 3),  # SEI, plating, capacity loss
            "thermal": nn.Linear(d_model, 2),      # Thermal resistance, heat gen
        })

    def forward(
        self,
        summary_vectors: torch.Tensor,
        prompt_input: Optional[torch.Tensor] = None,
        stress_context: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            summary_vectors: (B, n_summary, d_model) from encoder
            prompt_input: (B, L_prompt, d_model) optional prompt sequence
            stress_context: (B, n_stress, d_model) stress data for prompt
        Returns:
            dict with keys: soh, rul, degradation, thermal
        """
        # Build prompt
        x = self.prompt_processor(summary_vectors, stress_context)

        # Cross-attend to summary
        residual = x
        x_normed = self.cross_norm(x)
        x_cross, _ = self.cross_attention(x_normed, summary_vectors, summary_vectors)
        x = residual + x_cross

        # Add external prompt if provided
        if prompt_input is not None:
            x = torch.cat([x, prompt_input], dim=1)

        # Mamba-2 decoder layers
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # Global pool and produce outputs
        x_pooled = x.mean(dim=1)  # (B, d_model)

        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(x_pooled)

        # Constrain SOH to [0, 1]
        outputs["soh"] = torch.sigmoid(outputs["soh"])
        # RUL must be non-negative
        outputs["rul"] = F.softplus(outputs["rul"])

        return outputs

    def decode_step(
        self,
        summary_vectors: torch.Tensor,
        step_input: torch.Tensor,
    ) -> dict:
        """Single-step decoding for streaming inference."""
        x = step_input.unsqueeze(1) if step_input.dim() == 2 else step_input

        # Cross-attend to summary
        x_cross, _ = self.cross_attention(
            self.cross_norm(x), summary_vectors, summary_vectors
        )
        x = x + x_cross

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x_pooled = x.mean(dim=1)

        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(x_pooled)
        outputs["soh"] = torch.sigmoid(outputs["soh"])
        outputs["rul"] = F.softplus(outputs["rul"])
        return outputs
