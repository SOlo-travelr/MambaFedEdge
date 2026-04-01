"""
Physics-Informed Lightweight LSTM for on-device battery inference.

Designed for edge deployment on MCU/ECU (Cortex-R/M):
- Quantization-friendly (simple mul/add operations)
- Physics-aware gates encoding degradation dynamics
- Long-term memory for key events, short-term for recent updates
- INT8 compatible operations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class PhysicsGate(nn.Module):
    """Physics-informed gate that encodes electrochemical dynamics.

    Incorporates known battery degradation physics:
    - SEI layer growth (sqrt(t) dependence)
    - Lithium plating (exponential at low T / high C-rate)
    - Calendar aging (Arrhenius temperature dependence)
    """

    def __init__(self, d_hidden: int):
        super().__init__()
        self.d_hidden = d_hidden

        # Learnable physics parameters
        self.sei_rate = nn.Parameter(torch.tensor(0.01))
        self.plating_threshold = nn.Parameter(torch.tensor(0.1))
        self.arrhenius_factor = nn.Parameter(torch.tensor(0.05))

        self.physics_proj = nn.Linear(3, d_hidden, bias=False)

    def forward(
        self,
        temperature: Optional[torch.Tensor] = None,
        c_rate: Optional[torch.Tensor] = None,
        cycle_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute physics-based gating signal.

        Returns: (B, d_hidden) physics factor
        """
        B = temperature.shape[0] if temperature is not None else c_rate.shape[0]
        device = temperature.device if temperature is not None else c_rate.device

        physics_features = []

        # SEI growth factor: proportional to sqrt(cycle)
        if cycle_count is not None:
            sei = self.sei_rate * torch.sqrt(cycle_count.float() + 1.0)
        else:
            sei = torch.zeros(B, device=device)
        physics_features.append(sei)

        # Lithium plating risk: exponential at low-T / high C-rate
        if temperature is not None and c_rate is not None:
            plating = torch.exp(-temperature / 300.0) * torch.relu(c_rate - self.plating_threshold)
        else:
            plating = torch.zeros(B, device=device)
        physics_features.append(plating)

        # Calendar aging: Arrhenius-like
        if temperature is not None:
            cal_aging = self.arrhenius_factor * torch.exp(temperature / 300.0 - 1.0)
        else:
            cal_aging = torch.zeros(B, device=device)
        physics_features.append(cal_aging)

        physics = torch.stack(physics_features, dim=-1)  # (B, 3)
        return torch.sigmoid(self.physics_proj(physics))  # (B, d_hidden)


class PhysicsLSTM(nn.Module):
    """Physics-informed lightweight LSTM for edge BMS inference.

    Architecture:
    - Standard LSTM gates (input, forget, output, cell)
    - Physics-aware forget gate modulation
    - Quantization-friendly: primarily uses mul/add
    - Dual memory: long-term (slow-changing) + short-term (fast updates)
    """

    def __init__(
        self,
        d_input: int = 4,       # I, V, T, t
        d_hidden: int = 64,
        d_output: int = 1,      # SOH
        n_layers: int = 2,
        dropout: float = 0.1,
        use_physics_gate: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.n_layers = n_layers
        self.use_physics_gate = use_physics_gate
        self.bidirectional = bidirectional

        # Input normalization (batch norm is simpler for quantization)
        self.input_norm = nn.BatchNorm1d(d_input)

        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = d_hidden * (2 if bidirectional else 1)

        # Long-term memory gate
        self.long_term_gate = nn.Sequential(
            nn.Linear(lstm_out_dim + d_hidden, d_hidden),
            nn.Sigmoid(),
        )
        self.long_term_update = nn.Sequential(
            nn.Linear(lstm_out_dim, d_hidden),
            nn.Tanh(),
        )

        # Short-term memory (fast-track)
        self.short_term_proj = nn.Linear(lstm_out_dim, d_hidden)

        # Physics gate
        if use_physics_gate:
            self.physics_gate = PhysicsGate(d_hidden)

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_hidden * 2),  # long + short
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_output),
        )

        # Running long-term memory (registered as buffer for state tracking)
        self.register_buffer("_long_term_memory", torch.zeros(1, d_hidden))

    def forward(
        self,
        x: torch.Tensor,
        physics_data: Optional[Dict[str, torch.Tensor]] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        long_term_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (B, L, d_input) sensor data sequence
            physics_data: dict with temperature, c_rate, cycle_count
            hidden: optional LSTM hidden state
            long_term_memory: (B, d_hidden) external long-term memory
        Returns:
            output: (B, L, d_output) predictions per timestep
            state: dict with hidden, long_term_memory, short_term for continuation
        """
        B, L, D = x.shape

        # Normalize input
        x_flat = x.reshape(-1, D)
        x_flat = self.input_norm(x_flat)
        x = x_flat.reshape(B, L, D)

        # LSTM forward
        if hidden is not None:
            lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (B, L, d_hidden * n_dir)

        # Short-term memory (fast updates from recent data)
        short_term = self.short_term_proj(lstm_out)  # (B, L, d_hidden)

        # Long-term memory update (slow, retains key events)
        if long_term_memory is None:
            long_term_memory = self._long_term_memory.expand(B, -1)

        long_term_seq = []
        ltm = long_term_memory
        for t in range(L):
            lstm_t = lstm_out[:, t, :]  # (B, lstm_out_dim)
            # Gate controls how much new info enters long-term memory
            gate_input = torch.cat([lstm_t, ltm], dim=-1)
            gate = self.long_term_gate(gate_input)  # (B, d_hidden)
            update = self.long_term_update(lstm_t)   # (B, d_hidden)
            ltm = gate * ltm + (1 - gate) * update
            long_term_seq.append(ltm)

        long_term = torch.stack(long_term_seq, dim=1)  # (B, L, d_hidden)

        # Physics-aware modulation
        if self.use_physics_gate and physics_data is not None:
            temp = physics_data.get("temperature")
            crate = physics_data.get("c_rate")
            cycle = physics_data.get("cycle_count")
            physics_factor = self.physics_gate(temp, crate, cycle)  # (B, d_hidden)
            physics_factor = physics_factor.unsqueeze(1).expand(-1, L, -1)
            long_term = long_term * physics_factor

        # Combine memories for output
        combined = torch.cat([long_term, short_term], dim=-1)  # (B, L, d_hidden*2)
        output = self.output_head(combined)  # (B, L, d_output)

        state = {
            "hidden": (h_n, c_n),
            "long_term_memory": ltm,
            "short_term": short_term[:, -1, :],
        }

        return output, state

    def predict_step(
        self,
        x_step: torch.Tensor,
        state: Dict,
        physics_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Single-step inference for streaming on-device prediction.

        Args:
            x_step: (B, d_input) single timestep
            state: previous state dict
        Returns:
            output: (B, d_output)
            state: updated state dict
        """
        x_step = x_step.unsqueeze(1)  # (B, 1, d_input)
        output, state = self.forward(
            x_step,
            physics_data=physics_data,
            hidden=state.get("hidden"),
            long_term_memory=state.get("long_term_memory"),
        )
        return output.squeeze(1), state
