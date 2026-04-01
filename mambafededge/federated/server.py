"""
Federated Learning Server.

Coordinates federated training across edge devices:
- Distributes global model
- Collects and aggregates client updates
- Manages training rounds
- Supports multiple aggregation strategies
- Handles heterogeneous clients (different chemistries, hardware)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Callable
from collections import OrderedDict
import copy
import logging
import json
import os

from mambafededge.federated.aggregation import (
    federated_averaging,
    federated_prox,
    federated_scaffold,
    quality_weighted_aggregation,
)
from mambafededge.federated.client import FederatedClient

logger = logging.getLogger(__name__)


class FederatedServer:
    """Federated Learning Server for coordinating BMS model training."""

    def __init__(
        self,
        global_model: nn.Module,
        aggregation_strategy: str = "fedavg",
        n_rounds: int = 100,
        clients_per_round: Optional[int] = None,
        min_clients: int = 2,
        mu: float = 0.01,  # FedProx
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
    ):
        self.global_model = global_model.to(device)
        self.aggregation_strategy = aggregation_strategy
        self.n_rounds = n_rounds
        self.clients_per_round = clients_per_round
        self.min_clients = min_clients
        self.mu = mu
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience

        self.clients: Dict[str, FederatedClient] = {}
        self.round_history: List[Dict] = []
        self.best_loss = float("inf")
        self.patience_counter = 0

        # SCAFFOLD globals
        self._global_control: Optional[OrderedDict] = None

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def register_client(self, client: FederatedClient):
        """Register a new edge device client."""
        self.clients[client.client_id] = client
        logger.info(f"Registered client '{client.client_id}' (chemistry: {client.chemistry_type})")

    def unregister_client(self, client_id: str):
        """Remove a client."""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Unregistered client '{client_id}'")

    def get_global_state(self) -> OrderedDict:
        return copy.deepcopy(self.global_model.state_dict())

    def select_clients(self, n: Optional[int] = None) -> List[FederatedClient]:
        """Select clients for this round."""
        available = list(self.clients.values())
        if len(available) < self.min_clients:
            raise RuntimeError(
                f"Need at least {self.min_clients} clients, got {len(available)}"
            )

        n = n or self.clients_per_round or len(available)
        n = min(n, len(available))

        # Random selection
        indices = torch.randperm(len(available))[:n].tolist()
        return [available[i] for i in indices]

    def _aggregate(
        self,
        client_results: List[Dict[str, Any]],
    ) -> OrderedDict:
        """Aggregate client updates based on strategy."""
        global_state = self.get_global_state()
        client_states = [r["model_state"] for r in client_results]
        client_weights = [float(r["n_samples"]) for r in client_results]

        if self.aggregation_strategy == "fedavg":
            return federated_averaging(global_state, client_states, client_weights)

        elif self.aggregation_strategy == "fedprox":
            return federated_prox(global_state, client_states, client_weights, self.mu)

        elif self.aggregation_strategy == "scaffold":
            client_controls = [
                r.get("control_variate", global_state) or global_state
                for r in client_results
            ]
            if self._global_control is None:
                self._global_control = copy.deepcopy(global_state)

            new_state, new_control = federated_scaffold(
                global_state, client_states, client_controls, self._global_control, client_weights
            )
            self._global_control = new_control
            return new_state

        elif self.aggregation_strategy == "quality":
            client_metrics = [
                {"loss": r.get("val_loss", r.get("train_loss", 1.0))}
                for r in client_results
            ]
            return quality_weighted_aggregation(
                global_state, client_states, client_metrics
            )

        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")

    def train_round(self, round_num: int) -> Dict[str, Any]:
        """Execute one federated training round.

        Returns: dict with round metrics
        """
        logger.info(f"=== Round {round_num}/{self.n_rounds} ===")

        # Select clients
        selected_clients = self.select_clients()
        logger.info(f"Selected {len(selected_clients)} clients: "
                     f"{[c.client_id for c in selected_clients]}")

        # Distribute global model & collect local updates
        global_state = self.get_global_state()
        client_results = []

        for client in selected_clients:
            # Set SCAFFOLD control if needed
            if self.aggregation_strategy == "scaffold" and self._global_control is not None:
                client.set_global_control(self._global_control)

            result = client.local_train(
                global_state=global_state,
                mu=self.mu if self.aggregation_strategy == "fedprox" else 0.0,
            )
            client_results.append(result)
            logger.info(
                f"  Client '{result['client_id']}': "
                f"train_loss={result['train_loss']:.6f}, "
                f"val_loss={result.get('val_loss', 'N/A')}"
            )

        # Aggregate
        aggregated_state = self._aggregate(client_results)
        self.global_model.load_state_dict(aggregated_state)

        # Compute round metrics
        avg_train_loss = sum(r["train_loss"] for r in client_results) / len(client_results)
        val_losses = [r["val_loss"] for r in client_results if r.get("val_loss") is not None]
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else None

        round_info = {
            "round": round_num,
            "n_clients": len(selected_clients),
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "client_results": [
                {
                    "client_id": r["client_id"],
                    "chemistry": r["chemistry_type"],
                    "train_loss": r["train_loss"],
                    "val_loss": r.get("val_loss"),
                    "n_samples": r["n_samples"],
                }
                for r in client_results
            ],
        }
        self.round_history.append(round_info)

        # Early stopping check
        check_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss
        if check_loss < self.best_loss:
            self.best_loss = check_loss
            self.patience_counter = 0
            if self.checkpoint_dir:
                self._save_checkpoint(round_num, "best")
        else:
            self.patience_counter += 1

        logger.info(
            f"  Round {round_num} summary: "
            f"avg_train={avg_train_loss:.6f}, "
            f"avg_val={avg_val_loss if avg_val_loss else 'N/A'}, "
            f"best={self.best_loss:.6f}"
        )

        return round_info

    def train(
        self,
        n_rounds: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """Run full federated training.

        Args:
            n_rounds: override number of rounds
            callback: optional function called after each round
        Returns:
            list of round results
        """
        n = n_rounds or self.n_rounds
        results = []

        for round_num in range(1, n + 1):
            round_info = self.train_round(round_num)
            results.append(round_info)

            if callback:
                callback(round_info)

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at round {round_num}")
                break

            # Periodic checkpoint
            if self.checkpoint_dir and round_num % 10 == 0:
                self._save_checkpoint(round_num, f"round_{round_num}")

        return results

    def _save_checkpoint(self, round_num: int, tag: str):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "round": round_num,
            "model_state": self.global_model.state_dict(),
            "best_loss": self.best_loss,
        }, path)
        logger.info(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.global_model.load_state_dict(checkpoint["model_state"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        logger.info(f"Loaded checkpoint from {path}")

    def get_training_summary(self) -> Dict:
        """Get summary of training history."""
        if not self.round_history:
            return {"status": "no training performed"}

        return {
            "total_rounds": len(self.round_history),
            "best_loss": self.best_loss,
            "final_train_loss": self.round_history[-1]["avg_train_loss"],
            "final_val_loss": self.round_history[-1].get("avg_val_loss"),
            "total_clients": len(self.clients),
            "strategy": self.aggregation_strategy,
        }
