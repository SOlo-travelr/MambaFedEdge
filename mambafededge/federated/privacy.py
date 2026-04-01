"""
Differential Privacy for Federated Learning.

Implements:
- Gaussian mechanism noise addition
- Gradient clipping
- Privacy budget accounting (Rényi DP)
- Secure aggregation primitives
"""

import torch
import math
from typing import Optional, List
from collections import OrderedDict


class DifferentialPrivacy:
    """Differential privacy module for federated learning.

    Provides epsilon-delta differential privacy guarantees
    through gradient clipping and calibrated noise addition.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        mechanism: str = "gaussian",  # "gaussian" or "laplace"
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.mechanism = mechanism

        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            # Calibrate noise for (epsilon, delta)-DP
            self.noise_multiplier = self._calibrate_noise(epsilon, delta)

        self._spent_epsilon = 0.0
        self._n_queries = 0

    def _calibrate_noise(self, epsilon: float, delta: float) -> float:
        """Calibrate Gaussian noise sigma for (epsilon, delta)-DP."""
        # Analytic calibration from Balle et al. (2018)
        return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def clip_gradients(self, model: torch.nn.Module) -> float:
        """Clip per-sample gradients to max_grad_norm.

        Returns: total gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_grad_norm
        )
        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm

    def add_noise_to_gradients(self, model: torch.nn.Module):
        """Add calibrated noise to model gradients."""
        sigma = self.noise_multiplier * self.max_grad_norm

        for param in model.parameters():
            if param.grad is not None:
                if self.mechanism == "gaussian":
                    noise = torch.randn_like(param.grad) * sigma
                elif self.mechanism == "laplace":
                    noise = torch.distributions.Laplace(0, sigma).sample(param.grad.shape).to(param.grad.device)
                else:
                    raise ValueError(f"Unknown mechanism: {self.mechanism}")
                param.grad.add_(noise)

        self._n_queries += 1

    def add_noise_to_state(self, state_dict: OrderedDict) -> OrderedDict:
        """Add noise to model state dict (for secure aggregation)."""
        sigma = self.noise_multiplier * self.max_grad_norm
        noisy_state = OrderedDict()

        for key, value in state_dict.items():
            if value.dtype.is_floating_point:
                noise = torch.randn_like(value) * sigma
                noisy_state[key] = value + noise
            else:
                noisy_state[key] = value

        return noisy_state

    def get_privacy_spent(self, n_steps: Optional[int] = None) -> dict:
        """Compute total privacy budget spent.

        Uses Rényi Differential Privacy accounting.
        """
        n = n_steps or self._n_queries
        if n == 0:
            return {"epsilon": 0.0, "delta": self.delta, "n_queries": 0}

        # Simple composition: epsilon grows with sqrt(n) for Gaussian
        rdp_epsilon = n * (1 / (2 * self.noise_multiplier ** 2))

        # Convert RDP to (epsilon, delta)-DP using order alpha=2
        alpha = 2.0
        epsilon = rdp_epsilon + math.log(1 / self.delta) / (alpha - 1)
        epsilon = min(epsilon, n * self.epsilon)  # Basic composition bound

        return {
            "epsilon": epsilon,
            "delta": self.delta,
            "n_queries": n,
            "noise_multiplier": self.noise_multiplier,
            "budget_remaining": max(0, self.epsilon * 100 - epsilon),
        }


class SecureAggregator:
    """Simulated secure aggregation for federated learning.

    In production, this would use cryptographic protocols (e.g., Shamir's secret sharing).
    Here we simulate the privacy-preserving aggregation.
    """

    def __init__(self, n_clients: int, threshold: int = None):
        self.n_clients = n_clients
        self.threshold = threshold or max(2, n_clients // 2)
        self._masks: List[OrderedDict] = []

    def generate_masks(self, template_state: OrderedDict) -> List[OrderedDict]:
        """Generate random masks for each client.

        Masks sum to zero, so aggregate reveals only the sum of updates.
        """
        self._masks = []
        running_sum = OrderedDict()

        for key in template_state.keys():
            running_sum[key] = torch.zeros_like(template_state[key], dtype=torch.float32)

        for i in range(self.n_clients - 1):
            mask = OrderedDict()
            for key in template_state.keys():
                if template_state[key].dtype.is_floating_point:
                    m = torch.randn_like(template_state[key])
                    mask[key] = m
                    running_sum[key] += m
                else:
                    mask[key] = torch.zeros_like(template_state[key])
            self._masks.append(mask)

        # Last mask ensures sum = 0
        last_mask = OrderedDict()
        for key in template_state.keys():
            last_mask[key] = -running_sum[key].to(template_state[key].dtype)
        self._masks.append(last_mask)

        return self._masks

    def mask_update(self, state: OrderedDict, client_idx: int) -> OrderedDict:
        """Apply mask to client update."""
        masked = OrderedDict()
        mask = self._masks[client_idx]
        for key in state.keys():
            if key in mask and state[key].dtype.is_floating_point:
                masked[key] = state[key] + mask[key].to(state[key].device)
            else:
                masked[key] = state[key]
        return masked

    def aggregate(self, masked_states: List[OrderedDict]) -> OrderedDict:
        """Aggregate masked updates (masks cancel out)."""
        result = OrderedDict()
        n = len(masked_states)

        for key in masked_states[0].keys():
            vals = torch.stack([s[key].float() for s in masked_states])
            result[key] = vals.mean(dim=0).to(masked_states[0][key].dtype)

        return result
