"""
Battery Health Metrics.

Evaluation metrics for SOH, RUL, and degradation prediction.
"""

import torch
import numpy as np
from typing import Dict, Optional


class BatteryMetrics:
    """Comprehensive metrics for battery health prediction evaluation."""

    @staticmethod
    def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        return ((predictions - targets) ** 2).mean().item()

    @staticmethod
    def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        return ((predictions - targets) ** 2).mean().sqrt().item()

    @staticmethod
    def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        return (predictions - targets).abs().mean().item()

    @staticmethod
    def mape(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Mean Absolute Percentage Error."""
        mask = targets.abs() > 1e-8
        if mask.sum() == 0:
            return 0.0
        return (((predictions[mask] - targets[mask]) / targets[mask]).abs().mean() * 100).item()

    @staticmethod
    def r2_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """R² (coefficient of determination)."""
        ss_res = ((targets - predictions) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        return (1 - ss_res / (ss_tot + 1e-8)).item()

    @staticmethod
    def max_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        return (predictions - targets).abs().max().item()

    @staticmethod
    def compute_all(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all metrics."""
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        return {
            "mse": BatteryMetrics.mse(predictions, targets),
            "rmse": BatteryMetrics.rmse(predictions, targets),
            "mae": BatteryMetrics.mae(predictions, targets),
            "mape": BatteryMetrics.mape(predictions, targets),
            "r2": BatteryMetrics.r2_score(predictions, targets),
            "max_error": BatteryMetrics.max_error(predictions, targets),
        }

    @staticmethod
    def rul_score(
        predicted_rul: torch.Tensor,
        actual_rul: torch.Tensor,
    ) -> Dict[str, float]:
        """RUL-specific scoring (asymmetric: late predictions worse)."""
        error = predicted_rul - actual_rul
        # Penalize late predictions more (safety-critical)
        score = torch.where(
            error < 0,
            torch.exp(-error / 13) - 1,  # early
            torch.exp(error / 10) - 1,    # late (higher penalty)
        )
        return {
            "rul_score": score.mean().item(),
            "rul_mae": error.abs().mean().item(),
            "early_predictions": (error < 0).float().mean().item(),
            "late_predictions": (error > 0).float().mean().item(),
        }

    @staticmethod
    def uncertainty_calibration(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """Evaluate uncertainty calibration.

        A well-calibrated model should have errors proportional to uncertainty.
        """
        errors = (predictions - targets).abs()
        sorted_indices = uncertainties.argsort()

        bin_size = len(errors) // n_bins
        calibration_scores = []

        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(errors)
            bin_indices = sorted_indices[start:end]

            bin_errors = errors[bin_indices].mean().item()
            bin_unc = uncertainties[bin_indices].mean().item()
            calibration_scores.append(abs(bin_errors - bin_unc))

        return {
            "calibration_error": sum(calibration_scores) / n_bins,
            "mean_uncertainty": uncertainties.mean().item(),
            "mean_error": errors.mean().item(),
            "uncertainty_correlation": float(
                np.corrcoef(
                    uncertainties.cpu().numpy(),
                    errors.cpu().numpy()
                )[0, 1]
            ) if len(errors) > 2 else 0.0,
        }
