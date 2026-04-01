"""
Visualization utilities for MambaFedEdge.

Generates plots for:
- Training history (loss curves)
- SOH predictions vs ground truth
- Degradation trends
- Federated learning convergence
- Uncertainty visualization
"""

import os
from typing import Dict, List, Optional


def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[str] = None,
    title: str = "Training History",
):
    """Plot training and validation loss curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train Loss", color="blue")
    if history.get("val_loss") and any(v is not None for v in history["val_loss"]):
        val_losses = [v for v in history["val_loss"] if v is not None]
        ax.plot(range(len(val_losses)), val_losses, label="Val Loss", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1]
    if history.get("lr"):
        ax.plot(history["lr"], label="Learning Rate", color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()


def plot_soh_prediction(
    true_soh: list,
    pred_soh: list,
    uncertainty: Optional[list] = None,
    save_path: Optional[str] = None,
    title: str = "SOH Prediction",
):
    """Plot SOH prediction with optional uncertainty bands."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    cycles = list(range(len(true_soh)))
    ax.plot(cycles, true_soh, label="True SOH", color="blue", linewidth=2)
    ax.plot(cycles, pred_soh, label="Predicted SOH", color="red", linewidth=2, linestyle="--")

    if uncertainty is not None:
        pred_arr = np.array(pred_soh)
        unc_arr = np.array(uncertainty)
        ax.fill_between(
            cycles,
            pred_arr - 1.96 * unc_arr,
            pred_arr + 1.96 * unc_arr,
            alpha=0.2,
            color="red",
            label="95% CI",
        )

    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("State of Health", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()


def plot_federated_convergence(
    round_results: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Federated Learning Convergence",
):
    """Plot federated learning convergence across rounds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    rounds = [r["round"] for r in round_results]
    train_losses = [r["avg_train_loss"] for r in round_results]
    val_losses = [r.get("avg_val_loss") for r in round_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(rounds, train_losses, label="Avg Train Loss", marker="o", markersize=3)
    valid_val = [(r, v) for r, v in zip(rounds, val_losses) if v is not None]
    if valid_val:
        ax.plot(*zip(*valid_val), label="Avg Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per Round")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    n_clients = [r["n_clients"] for r in round_results]
    ax.bar(rounds, n_clients, alpha=0.7, color="steelblue")
    ax.set_xlabel("Round")
    ax.set_ylabel("Clients")
    ax.set_title("Clients per Round")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()
