"""
Mamba-2 Encoder for battery cycling data.

Encodes long battery cycling sequences into hidden state summary vectors.
Trained on chemistry-encoded cycler data under different operating conditions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from mambafededge.models.mamba2 import Mamba2Layer


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for temporal encoding of battery cycles."""

    def __init__(self, d_model: int, max_period: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, L) or (B, L, 1) time values
        """
        if t.dim() == 3:
            t = t.squeeze(-1)
        half = self.d_model // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32)
            * (torch.log(torch.tensor(self.max_period)) / half)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (B, L, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, L, d_model)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.linear(emb)


class StressEncoder(nn.Module):
    """Encodes battery stress factors: thermal, SOC, C-rate, rest, time stress."""

    STRESS_TYPES = ["time", "temperature", "soc", "rest", "crate"]

    def __init__(self, d_model: int):
        super().__init__()
        self.n_stresses = len(self.STRESS_TYPES)
        self.stress_proj = nn.Linear(self.n_stresses, d_model)
        self.norm = nn.LayerNorm(d_model)

    def compute_stresses(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute stress factors from raw sensor data.

        Args:
            data: dict with keys: voltage, current, temperature, time, soc
        Returns:
            (B, L, n_stresses) stress tensor
        """
        B, L = data["voltage"].shape[:2]
        device = data["voltage"].device
        stresses = []

        # Time stress: cumulative cycling time normalized
        if "time" in data:
            t = data["time"]
            time_stress = t / (t.max(dim=1, keepdim=True)[0] + 1e-8)
        else:
            time_stress = torch.linspace(0, 1, L, device=device).expand(B, L)
        stresses.append(time_stress)

        # Temperature stress: deviation from optimal (25°C / 298K)
        if "temperature" in data:
            temp = data["temperature"]
            temp_stress = torch.abs(temp - 298.0) / 50.0  # Normalized
        else:
            temp_stress = torch.zeros(B, L, device=device)
        stresses.append(temp_stress)

        # SOC stress: cycling depth
        if "soc" in data:
            soc = data["soc"]
            soc_stress = torch.abs(soc - 0.5) * 2.0  # Higher at extremes
        else:
            soc_stress = torch.zeros(B, L, device=device)
        stresses.append(soc_stress)

        # Rest stress: time since last rest period
        rest_stress = torch.zeros(B, L, device=device)
        stresses.append(rest_stress)

        # C-rate stress
        if "current" in data and "capacity" in data:
            crate = torch.abs(data["current"]) / (data["capacity"].unsqueeze(-1) + 1e-8)
            crate_stress = crate / 3.0  # Normalized by 3C
        elif "current" in data:
            crate_stress = torch.abs(data["current"]) / (torch.abs(data["current"]).max(dim=1, keepdim=True)[0] + 1e-8)
        else:
            crate_stress = torch.zeros(B, L, device=device)
        stresses.append(crate_stress)

        return torch.stack(stresses, dim=-1)  # (B, L, n_stresses)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        stresses = self.compute_stresses(data)
        return self.norm(self.stress_proj(stresses))


class Mamba2Encoder(nn.Module):
    """Mamba-2 Encoder for battery cycling data.

    Processes long sequences of battery data and produces:
    - Full hidden state sequence
    - Summary vectors (for decoder prompting)
    """

    def __init__(
        self,
        d_input: int = 4,           # raw sensor: I, V, T, t
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 6,
        n_heads: int = 4,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_time_embedding: bool = True,
        use_stress_encoding: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection from raw sensor data
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Time embedding
        self.use_time_embedding = use_time_embedding
        if use_time_embedding:
            self.time_embed = TimeEmbedding(d_model)

        # Stress encoding
        self.use_stress_encoding = use_stress_encoding
        if use_stress_encoding:
            self.stress_encoder = StressEncoder(d_model)
            self.stress_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )

        # Mamba-2 layers
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

        # Summary projection: produces fixed-size summary vectors
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # Attention pooling for summary
        self.summary_attention = nn.MultiheadAttention(
            d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.summary_queries = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)

    def forward(
        self,
        sensor_data: torch.Tensor,
        time_data: Optional[torch.Tensor] = None,
        stress_data: Optional[Dict[str, torch.Tensor]] = None,
        chemistry_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sensor_data: (B, L, d_input) raw sensor readings [I, V, T, t]
            time_data: (B, L) or (B, L, 1) timestamps
            stress_data: dict of sensor data for stress computation
            chemistry_embedding: (B, d_model) chemistry-aware embedding

        Returns:
            hidden_states: (B, L, d_model) full sequence hidden states
            summary_vectors: (B, n_summary, d_model) summary for decoder
        """
        B, L, _ = sensor_data.shape

        # Project input
        x = self.input_proj(sensor_data)  # (B, L, d_model)

        # Add time embedding
        if self.use_time_embedding and time_data is not None:
            x = x + self.time_embed(time_data)

        # Add stress encoding
        if self.use_stress_encoding and stress_data is not None:
            stress_emb = self.stress_encoder(stress_data)
            gate = self.stress_gate(torch.cat([x, stress_emb], dim=-1))
            x = x * gate + stress_emb * (1 - gate)

        # Add chemistry embedding (broadcast across sequence)
        if chemistry_embedding is not None:
            if chemistry_embedding.dim() == 2:
                chemistry_embedding = chemistry_embedding.unsqueeze(1).expand(-1, L, -1)
            x = x + chemistry_embedding

        # Process through Mamba-2 layers
        for layer in self.layers:
            x = layer(x)

        hidden_states = x  # (B, L, d_model)

        # Generate summary vectors via attention pooling
        queries = self.summary_queries.expand(B, -1, -1)
        summary, _ = self.summary_attention(queries, hidden_states, hidden_states)
        summary = self.summary_proj(summary)  # (B, n_summary, d_model)

        return hidden_states, summary

    def get_hidden_state_summary(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract summary from pre-computed hidden states."""
        B = hidden_states.shape[0]
        queries = self.summary_queries.expand(B, -1, -1)
        summary, _ = self.summary_attention(queries, hidden_states, hidden_states)
        return self.summary_proj(summary)
