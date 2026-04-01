"""
Uncertainty Quantification Heads.

Implements:
- MC Dropout: Monte Carlo Dropout for epistemic uncertainty
- Evidential Regression: Deep Evidential Regression for aleatoric + epistemic
- Ensemble-based uncertainty from federated model variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class MCDropoutHead(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation.

    During inference, keeps dropout active and runs multiple forward
    passes to estimate prediction mean and variance.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int = 1,
        d_hidden: int = 64,
        dropout_rate: float = 0.2,
        n_mc_samples: int = 30,
    ):
        super().__init__()
        self.n_mc_samples = n_mc_samples
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_hidden, d_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.layers(x)

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Run MC Dropout inference.

        Args:
            x: (B, d_input)
            n_samples: number of MC samples (default: self.n_mc_samples)
        Returns:
            dict with mean, std, samples
        """
        n = n_samples or self.n_mc_samples
        self.train()  # Keep dropout active

        samples = []
        for _ in range(n):
            pred = self.layers(x)
            samples.append(pred)

        samples = torch.stack(samples, dim=0)  # (n, B, d_output)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        return {
            "mean": mean,
            "std": std,
            "samples": samples,
            "confidence": 1.0 / (1.0 + std),
        }


class EvidentialHead(nn.Module):
    """Deep Evidential Regression for uncertainty quantification.

    Outputs parameters of a Normal-Inverse-Gamma distribution:
    (gamma, nu, alpha, beta) that encode both aleatoric and epistemic uncertainty.

    Reference: Amini et al., "Deep Evidential Regression" (NeurIPS 2020)
    """

    def __init__(self, d_input: int, d_output: int = 1, d_hidden: int = 64):
        super().__init__()
        self.d_output = d_output

        self.shared = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )

        # Output 4 parameters per output dimension
        self.gamma_head = nn.Linear(d_hidden, d_output)    # prediction mean
        self.nu_head = nn.Linear(d_hidden, d_output)       # epistemic confidence
        self.alpha_head = nn.Linear(d_hidden, d_output)    # aleatoric shape
        self.beta_head = nn.Linear(d_hidden, d_output)     # aleatoric scale

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with gamma (mean), nu, alpha, beta, aleatoric_uncertainty, epistemic_uncertainty
        """
        h = self.shared(x)

        gamma = self.gamma_head(h)
        nu = F.softplus(self.nu_head(h)) + 1e-6     # > 0
        alpha = F.softplus(self.alpha_head(h)) + 1.0  # > 1
        beta = F.softplus(self.beta_head(h)) + 1e-6  # > 0

        # Uncertainty decomposition
        aleatoric = beta / (alpha - 1.0 + 1e-8)  # Expected variance
        epistemic = aleatoric / (nu + 1e-8)       # Variance of the mean

        return {
            "mean": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": aleatoric + epistemic,
            "confidence": 1.0 / (1.0 + aleatoric + epistemic),
        }

    @staticmethod
    def evidential_loss(
        targets: torch.Tensor,
        gamma: torch.Tensor,
        nu: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        reg_coeff: float = 0.01,
    ) -> torch.Tensor:
        """Evidential regression loss (NIG loss).

        Args:
            targets: (B, d_output) ground truth
            gamma, nu, alpha, beta: NIG parameters
            reg_coeff: regularization coefficient for evidence
        """
        twoBlambda = 2 * beta * (1 + nu)

        nll = (
            0.5 * torch.log(math.pi / (nu + 1e-8))
            - alpha * torch.log(twoBlambda + 1e-8)
            + (alpha + 0.5) * torch.log(
                nu * (targets - gamma) ** 2 + twoBlambda + 1e-8
            )
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Evidence regularization
        evidence = 2 * nu + alpha
        reg = reg_coeff * torch.abs(targets - gamma) * evidence

        return (nll + reg).mean()


class UncertaintyHead(nn.Module):
    """Combined uncertainty estimation module.

    Selects between MC Dropout and Evidential Regression
    and provides a unified interface for uncertainty-aware predictions.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int = 1,
        d_hidden: int = 64,
        method: str = "evidential",  # "mc_dropout" or "evidential"
        dropout_rate: float = 0.2,
        n_mc_samples: int = 30,
        uncertainty_threshold: float = 0.3,
    ):
        super().__init__()
        self.method = method
        self.uncertainty_threshold = uncertainty_threshold

        if method == "mc_dropout":
            self.head = MCDropoutHead(
                d_input, d_output, d_hidden, dropout_rate, n_mc_samples
            )
        elif method == "evidential":
            self.head = EvidentialHead(d_input, d_output, d_hidden)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.method == "evidential":
            result = self.head(x)
        else:
            result = self.head.predict_with_uncertainty(x)
        return result

    def needs_fallback(self, uncertainty_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Determine if predictions need fallback system.

        Returns: (B,) boolean tensor indicating high uncertainty
        """
        if "total_uncertainty" in uncertainty_output:
            unc = uncertainty_output["total_uncertainty"]
        else:
            unc = uncertainty_output["std"]
        return (unc > self.uncertainty_threshold).any(dim=-1)

    def get_confidence_interval(
        self, uncertainty_output: Dict[str, torch.Tensor], confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get confidence interval for predictions."""
        mean = uncertainty_output["mean"]

        if "total_uncertainty" in uncertainty_output:
            std = torch.sqrt(uncertainty_output["total_uncertainty"])
        else:
            std = uncertainty_output["std"]

        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        lower = mean - z * std
        upper = mean + z * std
        return lower, upper
