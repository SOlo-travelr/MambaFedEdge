"""
Large Battery Model (LBM) - Full Mamba-2 Encoder-Decoder Pipeline.

Equivalent to an LLM but for battery domain:
- Encoder trained on chemistry-encoded cycler data
- Hidden state summary vectors link encoder to decoder
- Decoder provides high-accuracy fallback predictions
- Supports prompting from the Long Sequence Generator
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from mambafededge.models.encoder import Mamba2Encoder
from mambafededge.models.decoder import Mamba2Decoder
from mambafededge.models.chemistry_embedding import ChemistryEmbedding
from mambafededge.models.uncertainty import UncertaintyHead
from mambafededge.models.fallback import FallbackSystem
from mambafededge.models.prediction_heads import PredictionModule


class LongSequenceGenerator(nn.Module):
    """Long Sequence Generator / Prompter.

    Analyzes stored battery data to:
    1. Identify extreme stress points
    2. Generate structured prompts for the decoder
    3. Manage data storage and retrieval
    """

    def __init__(self, d_model: int = 128, n_extreme_points: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_extreme_points = n_extreme_points

        # Stress point scorer: identifies extreme events
        self.stress_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Prompt generator
        self.prompt_gen = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Temporal aggregation for long history
        self.temporal_pool = nn.AdaptiveAvgPool1d(n_extreme_points)

    def forward(
        self,
        hidden_states: torch.Tensor,
        stress_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate prompt from stored battery data.

        Args:
            hidden_states: (B, L, d_model) encoder hidden states (stored)
            stress_data: (B, L, d_model) stress embeddings
        Returns:
            prompt: (B, n_extreme, d_model)
        """
        B, L, D = hidden_states.shape

        # Score each timestep for stress extremity
        scores = self.stress_scorer(hidden_states).squeeze(-1)  # (B, L)

        # Select top-k extreme points
        k = min(self.n_extreme_points, L)
        _, top_indices = torch.topk(scores, k, dim=1)
        top_indices = top_indices.sort(dim=1)[0]  # chronological order

        # Gather extreme points
        extreme_points = torch.gather(
            hidden_states, 1, top_indices.unsqueeze(-1).expand(-1, -1, D)
        )

        # Add stress context if available
        if stress_data is not None:
            stress_points = torch.gather(
                stress_data, 1, top_indices.unsqueeze(-1).expand(-1, -1, D)
            )
            combined = torch.cat([extreme_points, stress_points], dim=-1)
        else:
            combined = torch.cat([extreme_points, extreme_points], dim=-1)

        prompt = self.prompt_gen(combined)
        return prompt


class HistoryLogger(nn.Module):
    """On-device history logging with compressed storage.

    Stores battery events in a compressed format using learned encoding.
    Supports:
    - Continuous ingestion of sensor data
    - Compression of old data (temporal pooling)
    - Fast retrieval for prompt generation
    """

    def __init__(self, d_model: int = 128, max_history: int = 10000, compression_ratio: int = 10):
        super().__init__()
        self.d_model = d_model
        self.max_history = max_history
        self.compression_ratio = compression_ratio

        # Compression encoder
        self.compressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Event importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Buffer for storing history
        self.register_buffer("_history", torch.zeros(1, max_history, d_model))
        self.register_buffer("_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_filled", torch.tensor(False, dtype=torch.bool))

    def append(self, new_data: torch.Tensor):
        """Append new data to history buffer.

        Args:
            new_data: (B, L, d_model) - uses first batch element
        """
        data = new_data[0]  # Take first batch element
        L = data.shape[0]
        ptr = self._ptr.item()

        for i in range(L):
            self._history[0, ptr] = data[i]
            ptr = (ptr + 1) % self.max_history
            if ptr == 0:
                self._filled.fill_(True)

        self._ptr.fill_(ptr)

    def get_history(self, n_recent: Optional[int] = None) -> torch.Tensor:
        """Retrieve stored history.

        Args:
            n_recent: number of recent entries (None = all)
        Returns:
            (1, L, d_model) history tensor
        """
        if self._filled:
            total = self.max_history
        else:
            total = self._ptr.item()

        if total == 0:
            return torch.zeros(1, 1, self.d_model, device=self._history.device)

        if n_recent is not None and n_recent < total:
            start = (self._ptr.item() - n_recent) % self.max_history
            indices = [(start + i) % self.max_history for i in range(n_recent)]
            return self._history[:, indices, :]
        else:
            if self._filled:
                # Reorder so oldest is first
                ptr = self._ptr.item()
                indices = list(range(ptr, self.max_history)) + list(range(ptr))
                return self._history[:, indices, :]
            else:
                return self._history[:, :total, :]

    def compress(self) -> torch.Tensor:
        """Compress history for efficient storage/transmission."""
        history = self.get_history()
        compressed = self.compressor(history)

        # Keep only important events
        importance = self.importance_scorer(history)
        compressed = compressed * importance
        return compressed


class LargeBatteryModel(nn.Module):
    """Large Battery Model (LBM) - Complete end-to-end pipeline.

    Architecture overview:
    ┌─────────────────────────────────────────────────┐
    │  Sensors → Chemistry Embedding → Mamba-2 Encoder │
    │                    ↓                              │
    │         Hidden State Summary Vectors              │
    │                    ↓                              │
    │  History Logger → Long Seq Generator → Prompt     │
    │                    ↓                              │
    │           Mamba-2 Decoder → Predictions           │
    │                    ↓                              │
    │  Uncertainty Head → Fallback System → Output      │
    └─────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        d_sensor: int = 4,
        d_model: int = 128,
        d_state: int = 64,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_fallback: bool = True,
        uncertainty_method: str = "evidential",
    ):
        super().__init__()
        self.d_model = d_model

        # Chemistry embedding
        self.chemistry_embed = ChemistryEmbedding(d_model=d_model)

        # Mamba-2 Encoder
        self.encoder = Mamba2Encoder(
            d_input=d_sensor,
            d_model=d_model,
            d_state=d_state,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Mamba-2 Decoder
        self.decoder = Mamba2Decoder(
            d_model=d_model,
            d_state=d_state,
            n_layers=decoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Long Sequence Generator
        self.seq_generator = LongSequenceGenerator(d_model=d_model)

        # History Logger
        self.history_logger = HistoryLogger(d_model=d_model)

        # Uncertainty head
        self.uncertainty = UncertaintyHead(
            d_input=d_model,
            d_output=1,
            method=uncertainty_method,
        )

        # Fallback system
        self.use_fallback = use_fallback
        if use_fallback:
            self.fallback = FallbackSystem(d_model=d_model)

        # Onboard prediction module (lightweight path)
        self.onboard_module = PredictionModule(
            d_sensor=d_sensor,
            d_model=d_model,
            uncertainty_method=uncertainty_method,
        )

    def encode(
        self,
        sensor_data: torch.Tensor,
        time_data: Optional[torch.Tensor] = None,
        chemistry_data: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode battery data (onsite/cloud training).

        Args:
            sensor_data: (B, L, d_sensor)
            time_data: (B, L) timestamps
            chemistry_data: dict from ChemistryEmbedding.encode_chemistry_name
        Returns:
            hidden_states: (B, L, d_model)
            summary_vectors: (B, n_summary, d_model)
        """
        chem_embed = None
        if chemistry_data is not None:
            chem_embed = self.chemistry_embed(**chemistry_data)

        hidden_states, summary_vectors = self.encoder(
            sensor_data=sensor_data,
            time_data=time_data,
            chemistry_embedding=chem_embed,
        )

        # Log to history
        self.history_logger.append(hidden_states.detach())

        return hidden_states, summary_vectors

    def decode(
        self,
        summary_vectors: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode from summary vectors (onboard inference).

        Args:
            summary_vectors: (B, n_summary, d_model)
            hidden_states: (B, L, d_model) for prompt generation
        """
        # Generate prompt from history if available
        prompt = None
        if hidden_states is not None:
            prompt = self.seq_generator(hidden_states)

        # Decoder forward
        outputs = self.decoder(
            summary_vectors=summary_vectors,
            prompt_input=prompt,
        )
        return outputs

    def forward(
        self,
        sensor_data: torch.Tensor,
        time_data: Optional[torch.Tensor] = None,
        chemistry_data: Optional[Dict] = None,
        physics_inputs: Optional[Dict[str, torch.Tensor]] = None,
        use_onboard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through LBM.

        Args:
            sensor_data: (B, L, d_sensor) raw sensor readings
            time_data: (B, L) timestamps
            chemistry_data: chemistry info dict
            physics_inputs: dict for physics fallback
            use_onboard: if True, use lightweight onboard module
        """
        if use_onboard:
            # Lightweight onboard path
            chem_embed = None
            if chemistry_data is not None:
                chem_embed = self.chemistry_embed(**chemistry_data)
            return self.onboard_module(
                sensor_data=sensor_data,
                chemistry_embedding=chem_embed,
                physics_data=physics_inputs,
            )

        # Full LBM path
        hidden_states, summary_vectors = self.encode(
            sensor_data, time_data, chemistry_data
        )

        outputs = self.decode(summary_vectors, hidden_states)

        # Add uncertainty
        pooled = hidden_states.mean(dim=1)
        unc_output = self.uncertainty(pooled)
        outputs["uncertainty"] = unc_output

        # Fallback check
        if self.use_fallback and physics_inputs is not None:
            outputs = self.fallback(
                primary_prediction=outputs,
                features=pooled,
                physics_inputs=physics_inputs,
            )

        return outputs

    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "encoder_parameters": encoder_params,
            "decoder_parameters": decoder_params,
            "model_size_mb": total * 4 / (1024 * 1024),  # float32
            "quantized_size_mb": total * 1 / (1024 * 1024),  # int8
        }
