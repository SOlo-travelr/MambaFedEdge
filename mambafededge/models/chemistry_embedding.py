"""
Chemistry-Aware Embedding Module.

Encodes battery chemistry information (NMC, LFP, NCA, SSB, etc.)
and electrochemical properties into dense embeddings that condition
the entire prediction pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# Known battery chemistry types with properties
CHEMISTRY_DB = {
    "NMC111": {"cathode": "NMC", "ratio": [1, 1, 1], "voltage_nom": 3.6, "energy_density": 200, "cycle_life": 1000, "thermal_stability": 0.6},
    "NMC532": {"cathode": "NMC", "ratio": [5, 3, 2], "voltage_nom": 3.65, "energy_density": 220, "cycle_life": 1500, "thermal_stability": 0.65},
    "NMC622": {"cathode": "NMC", "ratio": [6, 2, 2], "voltage_nom": 3.65, "energy_density": 240, "cycle_life": 1200, "thermal_stability": 0.6},
    "NMC811": {"cathode": "NMC", "ratio": [8, 1, 1], "voltage_nom": 3.7, "energy_density": 280, "cycle_life": 800, "thermal_stability": 0.5},
    "LFP":    {"cathode": "LFP", "ratio": [1, 1, 1], "voltage_nom": 3.2, "energy_density": 160, "cycle_life": 3000, "thermal_stability": 0.95},
    "NCA":    {"cathode": "NCA", "ratio": [8, 1.5, 0.5], "voltage_nom": 3.65, "energy_density": 260, "cycle_life": 1000, "thermal_stability": 0.55},
    "LCO":    {"cathode": "LCO", "ratio": [1, 0, 0], "voltage_nom": 3.7, "energy_density": 200, "cycle_life": 500, "thermal_stability": 0.4},
    "LMO":    {"cathode": "LMO", "ratio": [0, 1, 0], "voltage_nom": 3.7, "energy_density": 150, "cycle_life": 700, "thermal_stability": 0.7},
    "SSB":    {"cathode": "SSB", "ratio": [1, 1, 1], "voltage_nom": 3.8, "energy_density": 350, "cycle_life": 2000, "thermal_stability": 0.9},
}

CATHODE_TYPES = ["NMC", "LFP", "NCA", "LCO", "LMO", "SSB", "OTHER"]


class ChemistryEmbedding(nn.Module):
    """Chemistry-aware input embedding for battery data conditioning.

    Supports:
    - Discrete chemistry type encoding (categorical)
    - Continuous electrochemical property encoding
    - Mixed embedding from both sources
    - Few-shot adaptation to new chemistries via property encoding
    """

    def __init__(
        self,
        d_model: int = 128,
        n_chemistry_types: int = len(CHEMISTRY_DB) + 1,  # +1 for unknown
        n_properties: int = 6,  # voltage, energy density, cycle life, thermal, capacity, impedance
        use_property_encoding: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_property_encoding = use_property_encoding

        # Categorical chemistry embedding
        self.chem_type_embed = nn.Embedding(n_chemistry_types, d_model)
        # Cathode family embedding
        self.cathode_embed = nn.Embedding(len(CATHODE_TYPES), d_model // 4)

        # Continuous property encoder
        if use_property_encoding:
            self.property_encoder = nn.Sequential(
                nn.Linear(n_properties, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model),
            )

        # Composition ratio encoder (e.g., Ni:Mn:Co ratios)
        self.ratio_encoder = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2),
        )

        # Fusion layer
        fusion_input_dim = d_model  # type embed
        if use_property_encoding:
            fusion_input_dim += d_model  # property embed
        fusion_input_dim += d_model // 4  # cathode embed
        fusion_input_dim += d_model // 2  # ratio embed

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Chemistry type to index mapping
        self._chem_to_idx = {name: i for i, name in enumerate(CHEMISTRY_DB.keys())}
        self._chem_to_idx["unknown"] = len(CHEMISTRY_DB)

    def get_chemistry_index(self, chemistry_name: str) -> int:
        return self._chem_to_idx.get(chemistry_name, self._chem_to_idx["unknown"])

    def get_cathode_index(self, chemistry_name: str) -> int:
        info = CHEMISTRY_DB.get(chemistry_name, {})
        cathode = info.get("cathode", "OTHER")
        return CATHODE_TYPES.index(cathode) if cathode in CATHODE_TYPES else CATHODE_TYPES.index("OTHER")

    def encode_chemistry_name(self, names: list, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert chemistry names to all required tensors."""
        indices = torch.tensor([self.get_chemistry_index(n) for n in names], device=device)
        cathode_indices = torch.tensor([self.get_cathode_index(n) for n in names], device=device)

        properties = []
        ratios = []
        for name in names:
            info = CHEMISTRY_DB.get(name, CHEMISTRY_DB["NMC111"])
            properties.append([
                info["voltage_nom"] / 4.0,  # normalize
                info["energy_density"] / 400.0,
                info["cycle_life"] / 5000.0,
                info["thermal_stability"],
                0.5,  # default capacity factor
                0.5,  # default impedance factor
            ])
            ratios.append([r / 10.0 for r in info["ratio"]])

        return {
            "chem_idx": indices,
            "cathode_idx": cathode_indices,
            "properties": torch.tensor(properties, dtype=torch.float32, device=device),
            "ratios": torch.tensor(ratios, dtype=torch.float32, device=device),
        }

    def forward(
        self,
        chem_idx: Optional[torch.Tensor] = None,
        cathode_idx: Optional[torch.Tensor] = None,
        properties: Optional[torch.Tensor] = None,
        ratios: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            chem_idx: (B,) chemistry type indices
            cathode_idx: (B,) cathode family indices
            properties: (B, n_properties) continuous electrochemical properties
            ratios: (B, 3) composition ratios (e.g., Ni:Mn:Co)
        Returns:
            (B, d_model) chemistry embedding
        """
        B = chem_idx.shape[0] if chem_idx is not None else properties.shape[0]
        device = chem_idx.device if chem_idx is not None else properties.device

        parts = []

        # Chemistry type embedding
        if chem_idx is not None:
            parts.append(self.chem_type_embed(chem_idx))
        else:
            parts.append(torch.zeros(B, self.d_model, device=device))

        # Property encoding
        if self.use_property_encoding and properties is not None:
            parts.append(self.property_encoder(properties))
        elif self.use_property_encoding:
            parts.append(torch.zeros(B, self.d_model, device=device))

        # Cathode embedding
        if cathode_idx is not None:
            parts.append(self.cathode_embed(cathode_idx))
        else:
            parts.append(torch.zeros(B, self.d_model // 4, device=device))

        # Ratio embedding
        if ratios is not None:
            parts.append(self.ratio_encoder(ratios))
        else:
            parts.append(torch.zeros(B, self.d_model // 2, device=device))

        # Fuse all embeddings
        combined = torch.cat(parts, dim=-1)
        return self.fusion(combined)
