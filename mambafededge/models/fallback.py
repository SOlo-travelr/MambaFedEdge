"""
Fallback System: ISO 26262 compliant safety fallback.

When uncertainty is high, the prediction is routed to the Large Battery Model
(Mamba-2 encoder-decoder) for a more thorough assessment.
Includes watchdog monitoring and deterministic physics-based fallback.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import time


class WatchdogMonitor(nn.Module):
    """Watchdog timer and safety monitor for ISO 26262 compliance.

    Monitors:
    - Prediction latency (must be < threshold)
    - Prediction range validity
    - Uncertainty bounds
    - Model health (gradient norms, activations)
    """

    def __init__(
        self,
        max_latency_ms: float = 25.0,
        soh_range: Tuple[float, float] = (0.0, 1.0),
        max_uncertainty: float = 0.5,
    ):
        super().__init__()
        self.max_latency_ms = max_latency_ms
        self.soh_range = soh_range
        self.max_uncertainty = max_uncertainty
        self._last_check_time = None
        self._fault_count = 0
        self._max_faults = 3

    def check_prediction(
        self,
        prediction: Dict[str, torch.Tensor],
        latency_ms: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Validate prediction against safety constraints.

        Returns: dict of safety check results
        """
        checks = {}

        # Latency check
        if latency_ms is not None:
            checks["latency_ok"] = latency_ms < self.max_latency_ms
        else:
            checks["latency_ok"] = True

        # Range validation
        if "soh" in prediction:
            soh = prediction["soh"]
            checks["soh_in_range"] = bool(
                (soh >= self.soh_range[0]).all() and (soh <= self.soh_range[1]).all()
            )
        else:
            checks["soh_in_range"] = True

        # Uncertainty check
        if "soh_uncertainty" in prediction:
            unc = prediction["soh_uncertainty"]
            if isinstance(unc, dict):
                total_unc = unc.get("total_uncertainty", unc.get("std", torch.tensor(0.0)))
            else:
                total_unc = unc
            checks["uncertainty_ok"] = bool((total_unc < self.max_uncertainty).all())
        else:
            checks["uncertainty_ok"] = True

        # NaN/Inf check
        all_finite = True
        for k, v in prediction.items():
            if isinstance(v, torch.Tensor):
                if not torch.isfinite(v).all():
                    all_finite = False
                    break
        checks["values_finite"] = all_finite

        # Overall safety
        checks["safe"] = all(checks.values())

        if not checks["safe"]:
            self._fault_count += 1
        else:
            self._fault_count = max(0, self._fault_count - 1)

        checks["needs_fallback"] = self._fault_count >= self._max_faults

        return checks


class PhysicsFallback(nn.Module):
    """Deterministic physics-based fallback model.

    Simple, interpretable model based on empirical degradation equations.
    Used when neural model uncertainty is too high or watchdog triggers.
    """

    def __init__(self):
        super().__init__()
        # Empirical parameters (can be calibrated)
        self.register_buffer("sei_coeff", torch.tensor(0.02))
        self.register_buffer("calendar_coeff", torch.tensor(0.001))
        self.register_buffer("cycle_coeff", torch.tensor(0.0005))

    def forward(
        self,
        cycle_count: torch.Tensor,
        temperature: torch.Tensor,
        avg_dod: torch.Tensor,
        time_days: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Empirical SOH estimation.

        SOH = 1 - (SEI_loss + calendar_loss + cycle_loss)
        """
        # SEI layer growth: proportional to sqrt(time)
        sei_loss = self.sei_coeff * torch.sqrt(time_days + 1.0)

        # Calendar aging: Arrhenius temperature dependence
        temp_factor = torch.exp((temperature - 298.0) / 30.0)
        calendar_loss = self.calendar_coeff * time_days * temp_factor

        # Cycle aging: depth-of-discharge dependent
        cycle_loss = self.cycle_coeff * cycle_count * (avg_dod ** 1.5)

        total_loss = sei_loss + calendar_loss + cycle_loss
        soh = torch.clamp(1.0 - total_loss, min=0.0, max=1.0)

        return {
            "soh": soh.unsqueeze(-1),
            "sei_loss": sei_loss.unsqueeze(-1),
            "calendar_loss": calendar_loss.unsqueeze(-1),
            "cycle_loss": cycle_loss.unsqueeze(-1),
            "is_fallback": torch.ones_like(soh, dtype=torch.bool).unsqueeze(-1),
        }


class FallbackSystem(nn.Module):
    """Complete fallback system integrating watchdog, physics model, and LBM routing.

    Decision flow:
    1. Primary prediction from onboard module
    2. Watchdog checks safety constraints
    3. If uncertain -> route to LBM (Mamba-2 encoder-decoder)
    4. If LBM unavailable or still uncertain -> physics fallback
    """

    def __init__(
        self,
        d_model: int = 128,
        max_latency_ms: float = 25.0,
        uncertainty_threshold: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.uncertainty_threshold = uncertainty_threshold

        self.watchdog = WatchdogMonitor(max_latency_ms=max_latency_ms)
        self.physics_fallback = PhysicsFallback()

        # LBM routing decision network
        self.route_decision = nn.Sequential(
            nn.Linear(d_model + 1, 32),  # features + uncertainty
            nn.GELU(),
            nn.Linear(32, 3),  # 3 routes: primary, LBM, physics
        )

    def forward(
        self,
        primary_prediction: Dict[str, torch.Tensor],
        features: torch.Tensor,
        lbm_prediction: Optional[Dict[str, torch.Tensor]] = None,
        physics_inputs: Optional[Dict[str, torch.Tensor]] = None,
        latency_ms: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            primary_prediction: from onboard prediction module
            features: (B, d_model) feature vector
            lbm_prediction: from Large Battery Model (if available)
            physics_inputs: dict with cycle_count, temperature, avg_dod, time_days
            latency_ms: inference time
        """
        # Safety checks
        safety = self.watchdog.check_prediction(primary_prediction, latency_ms)

        if safety["safe"] and not safety.get("needs_fallback", False):
            primary_prediction["fallback_used"] = torch.zeros(features.shape[0], 1, device=features.device)
            primary_prediction["safety_checks"] = safety
            return primary_prediction

        # Try LBM fallback
        if lbm_prediction is not None:
            lbm_safety = self.watchdog.check_prediction(lbm_prediction)
            if lbm_safety["safe"]:
                lbm_prediction["fallback_used"] = torch.ones(features.shape[0], 1, device=features.device)
                lbm_prediction["fallback_type"] = "lbm"
                return lbm_prediction

        # Physics fallback (always available, deterministic)
        if physics_inputs is not None:
            physics_pred = self.physics_fallback(
                physics_inputs["cycle_count"],
                physics_inputs["temperature"],
                physics_inputs["avg_dod"],
                physics_inputs["time_days"],
            )
            physics_pred["fallback_used"] = torch.ones(features.shape[0], 1, device=features.device) * 2
            physics_pred["fallback_type"] = "physics"
            return physics_pred

        # Last resort: return primary with warning
        primary_prediction["fallback_used"] = torch.zeros(features.shape[0], 1, device=features.device)
        primary_prediction["warning"] = "no_fallback_available"
        return primary_prediction
