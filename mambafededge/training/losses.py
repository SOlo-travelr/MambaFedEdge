"""
Physics-Informed Losses for Battery Health Prediction.

Combines data-driven loss with physics-based constraints:
- SOH monotonicity (health can only decrease over time)
- Physics degradation consistency
- Uncertainty calibration
- Multi-task weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss for SOH/RUL prediction.

    L_total = L_data + λ_mono * L_monotonicity + λ_phys * L_physics + λ_unc * L_uncertainty
    """

    def __init__(
        self,
        lambda_monotonicity: float = 0.1,
        lambda_physics: float = 0.05,
        lambda_uncertainty: float = 0.01,
        lambda_smoothness: float = 0.01,
    ):
        super().__init__()
        self.lambda_mono = lambda_monotonicity
        self.lambda_phys = lambda_physics
        self.lambda_unc = lambda_uncertainty
        self.lambda_smooth = lambda_smoothness

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        predictions_seq: Optional[torch.Tensor] = None,
        physics_constraints: Optional[Dict[str, torch.Tensor]] = None,
        uncertainty_output: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: (B,) or (B, 1) predicted SOH
            targets: (B,) or (B, 1) true SOH
            predictions_seq: (B, L) sequential predictions for monotonicity
            physics_constraints: dict with physics losses
            uncertainty_output: dict from uncertainty head
        Returns:
            dict with total_loss and component losses
        """
        # Ensure matching shapes
        predictions = predictions.squeeze()
        targets = targets.squeeze()

        losses = {}

        # Data loss (MSE)
        losses["mse"] = F.mse_loss(predictions, targets)
        losses["mae"] = F.l1_loss(predictions, targets)

        total = losses["mse"]

        # Monotonicity constraint: SOH should not increase over time
        if predictions_seq is not None and predictions_seq.shape[1] > 1:
            diffs = predictions_seq[:, 1:] - predictions_seq[:, :-1]
            mono_violation = F.relu(diffs)  # penalize increases
            losses["monotonicity"] = mono_violation.mean()
            total = total + self.lambda_mono * losses["monotonicity"]

        # Physics consistency
        if physics_constraints is not None:
            phys_loss = torch.tensor(0.0, device=predictions.device)
            for name, constraint in physics_constraints.items():
                if isinstance(constraint, torch.Tensor):
                    phys_loss = phys_loss + constraint.mean()
            losses["physics"] = phys_loss
            total = total + self.lambda_phys * phys_loss

        # Uncertainty calibration loss
        if uncertainty_output is not None:
            if "gamma" in uncertainty_output:
                # Evidential regression loss
                from mambafededge.models.uncertainty import EvidentialHead
                unc_loss = EvidentialHead.evidential_loss(
                    targets.unsqueeze(-1) if targets.dim() == 1 else targets,
                    uncertainty_output["gamma"],
                    uncertainty_output["nu"],
                    uncertainty_output["alpha"],
                    uncertainty_output["beta"],
                )
                losses["uncertainty"] = unc_loss
                total = total + self.lambda_unc * unc_loss

        # Smoothness regularization
        if predictions_seq is not None and predictions_seq.shape[1] > 2:
            second_diff = predictions_seq[:, 2:] - 2 * predictions_seq[:, 1:-1] + predictions_seq[:, :-2]
            losses["smoothness"] = (second_diff ** 2).mean()
            total = total + self.lambda_smooth * losses["smoothness"]

        losses["total"] = total
        return losses


class CombinedBMSLoss(nn.Module):
    """Combined multi-task loss for BMS prediction.

    Handles multiple output heads with automatic weighting.
    """

    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        use_uncertainty_weighting: bool = True,
    ):
        super().__init__()
        default_weights = {
            "soh": 1.0,
            "rul": 0.5,
            "degradation": 0.3,
            "thermal": 0.2,
            "anomaly": 0.1,
        }
        self.task_weights = task_weights or default_weights
        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            # Learnable task-specific uncertainty (Kendall et al.)
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1))
                for task in self.task_weights.keys()
            })

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: dict of predictions per task
            targets: dict of targets per task
        """
        losses = {}
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)

        for task, weight in self.task_weights.items():
            if task in predictions and task in targets:
                pred = predictions[task].squeeze()
                targ = targets[task].squeeze()

                task_loss = F.mse_loss(pred, targ)
                losses[f"{task}_loss"] = task_loss

                if self.use_uncertainty_weighting and task in self.log_vars:
                    precision = torch.exp(-self.log_vars[task])
                    weighted_loss = precision * task_loss + self.log_vars[task]
                else:
                    weighted_loss = weight * task_loss

                total = total + weighted_loss

        losses["total"] = total
        return losses
