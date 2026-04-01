"""
Prediction Module: Multi-head output for BMS control.

Produces multiple outputs from a shared backbone:
- SOH for balancing logic
- Degradation parameters for DFET/CFET gate control
- Thermal resistance updates
- Variable C-rate recommendations
- Anomaly/fault detection scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from mambafededge.models.uncertainty import UncertaintyHead


class MultiHeadOutput(nn.Module):
    """Multi-head prediction module for BMS outputs.

    Each head produces a specific output with its own uncertainty estimate.
    """

    def __init__(self, d_input: int = 128, d_hidden: int = 64, uncertainty_method: str = "evidential"):
        super().__init__()

        # Shared backbone
        self.shared = nn.Sequential(
            nn.LayerNorm(d_input),
            nn.Linear(d_input, d_hidden * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
        )

        # SOH head: State of Health [0, 1]
        self.soh_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid(),
        )

        # RUL head: Remaining Useful Life (cycles)
        self.rul_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Softplus(),
        )

        # Degradation head: SEI growth, Li plating, capacity fade
        self.degradation_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 3),
            nn.Sigmoid(),  # [0, 1] normalized rates
        )

        # Thermal head: thermal resistance & heat generation rate
        self.thermal_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 2),
            nn.Softplus(),
        )

        # C-rate recommendation head
        self.crate_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid(),  # normalized C-rate [0, 1] -> mapped to actual range
        )

        # Anomaly detection head (latent space anomaly)
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid(),  # anomaly score [0, 1]
        )

        # Uncertainty head for main SOH prediction
        self.uncertainty = UncertaintyHead(
            d_input=d_hidden,
            d_output=1,
            d_hidden=d_hidden // 2,
            method=uncertainty_method,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, d_input) feature vector from backbone
        Returns:
            dict with all prediction outputs and uncertainty
        """
        h = self.shared(x)  # (B, d_hidden)

        outputs = {
            "soh": self.soh_head(h),
            "rul": self.rul_head(h),
            "degradation": self.degradation_head(h),  # (B, 3): SEI, plating, fade
            "thermal": self.thermal_head(h),           # (B, 2): resistance, heat_gen
            "crate_recommendation": self.crate_head(h) * 3.0,  # Scale to [0, 3C]
            "anomaly_score": self.anomaly_head(h),
        }

        # Uncertainty for SOH
        uncertainty_output = self.uncertainty(h)
        outputs["soh_uncertainty"] = uncertainty_output

        return outputs


class PredictionModule(nn.Module):
    """Complete onboard prediction module.

    Integrates chemistry-aware embedding, sensor input processing,
    lightweight LSTM backbone, and multi-head output.
    """

    def __init__(
        self,
        d_sensor: int = 4,          # I, V, T, t
        d_model: int = 128,
        d_hidden: int = 64,
        n_lstm_layers: int = 2,
        use_chemistry: bool = True,
        use_physics: bool = True,
        uncertainty_method: str = "evidential",
    ):
        super().__init__()
        self.d_model = d_model
        self.use_chemistry = use_chemistry

        # Sensor input projection
        self.sensor_proj = nn.Sequential(
            nn.Linear(d_sensor, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Chemistry embedding fusion
        if use_chemistry:
            self.chemistry_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )

        # Lightweight LSTM backbone
        from mambafededge.models.physics_lstm import PhysicsLSTM
        self.backbone = PhysicsLSTM(
            d_input=d_model,
            d_hidden=d_hidden,
            d_output=d_hidden,
            n_layers=n_lstm_layers,
            use_physics_gate=use_physics,
        )

        # Multi-head output
        self.output_module = MultiHeadOutput(
            d_input=d_hidden,
            d_hidden=d_hidden,
            uncertainty_method=uncertainty_method,
        )

    def forward(
        self,
        sensor_data: torch.Tensor,
        chemistry_embedding: Optional[torch.Tensor] = None,
        physics_data: Optional[Dict[str, torch.Tensor]] = None,
        state: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            sensor_data: (B, L, d_sensor) raw sensor readings
            chemistry_embedding: (B, d_model) chemistry conditioning
            physics_data: dict for physics gate
            state: previous LSTM state for streaming
        Returns:
            dict with all predictions
        """
        B, L, _ = sensor_data.shape

        # Project sensor data
        x = self.sensor_proj(sensor_data)  # (B, L, d_model)

        # Fuse chemistry embedding
        if self.use_chemistry and chemistry_embedding is not None:
            chem = chemistry_embedding.unsqueeze(1).expand(-1, L, -1)
            gate = self.chemistry_gate(torch.cat([x, chem], dim=-1))
            x = gate * x + (1 - gate) * chem

        # LSTM backbone
        lstm_out, new_state = self.backbone(x, physics_data=physics_data)
        # lstm_out: (B, L, d_hidden)

        # Use last timestep for prediction
        features = lstm_out[:, -1, :]  # (B, d_hidden)

        # Multi-head prediction
        outputs = self.output_module(features)
        outputs["_state"] = new_state

        return outputs
