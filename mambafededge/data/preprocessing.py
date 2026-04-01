"""
Battery Data Preprocessing.

Signal processing and feature engineering for battery cycling data.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List


class BatteryDataPreprocessor:
    """Preprocessing pipeline for battery cycling data."""

    def __init__(
        self,
        normalize: bool = True,
        compute_features: bool = True,
        window_size: int = 10,
    ):
        self.normalize = normalize
        self.compute_features = compute_features
        self.window_size = window_size
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None

    def fit(self, data: torch.Tensor):
        """Compute normalization statistics."""
        self._mean = data.mean(dim=0)
        self._std = data.std(dim=0) + 1e-8

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing."""
        if self.normalize and self._mean is not None:
            data = (data - self._mean) / self._std

        if self.compute_features:
            data = self._add_features(data)

        return data

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)

    def _add_features(self, data: torch.Tensor) -> torch.Tensor:
        """Add engineered features: rolling stats, derivatives, etc."""
        features = [data]

        # Rolling mean (simplified for 2D tensors)
        if data.dim() == 2 and data.shape[0] > self.window_size:
            roll_mean = torch.zeros_like(data)
            for i in range(data.shape[0]):
                start = max(0, i - self.window_size)
                roll_mean[i] = data[start:i+1].mean(dim=0)
            features.append(roll_mean)

        # First derivative (finite difference)
        if data.shape[0] > 1:
            deriv = torch.zeros_like(data)
            deriv[1:] = data[1:] - data[:-1]
            features.append(deriv)

        return torch.cat(features, dim=-1)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Reverse normalization."""
        if self._mean is not None:
            n_orig = self._mean.shape[0]
            return data[..., :n_orig] * self._std + self._mean
        return data

    def extract_cycle_features(
        self, current: np.ndarray, voltage: np.ndarray, temperature: np.ndarray
    ) -> Dict[str, float]:
        """Extract per-cycle features from raw data."""
        features = {}

        # Capacity (Ah)
        dt = 1.0  # assume 1s sampling
        features["capacity"] = np.abs(np.trapz(current, dx=dt)) / 3600

        # Energy (Wh)
        features["energy"] = np.abs(np.trapz(current * voltage, dx=dt)) / 3600

        # IC/DV features
        dV = np.diff(voltage)
        dQ = np.abs(current[:-1]) * dt / 3600
        valid = np.abs(dV) > 1e-6
        if valid.sum() > 0:
            ic = dQ[valid] / np.abs(dV[valid])
            features["ic_peak"] = float(ic.max())
            features["ic_mean"] = float(ic.mean())

        # Temperature stats
        features["temp_mean"] = float(np.mean(temperature))
        features["temp_max"] = float(np.max(temperature))
        features["temp_range"] = float(np.max(temperature) - np.min(temperature))

        # Voltage stats
        features["voltage_mean"] = float(np.mean(voltage))
        features["voltage_range"] = float(np.max(voltage) - np.min(voltage))

        return features
