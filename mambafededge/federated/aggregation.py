"""
Federated Learning Aggregation Strategies.

Implements:
- FedAvg: Federated Averaging
- FedProx: Proximal term regularization
- SCAFFOLD: Control variates for variance reduction
- Weighted aggregation based on data quality
"""

import torch
import copy
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict


def federated_averaging(
    global_model_state: OrderedDict,
    client_states: List[OrderedDict],
    client_weights: Optional[List[float]] = None,
) -> OrderedDict:
    """Federated Averaging (FedAvg).

    Aggregates model updates from multiple clients using weighted average.

    Args:
        global_model_state: current global model state dict
        client_states: list of client model state dicts
        client_weights: optional weights per client (e.g., proportional to data size)
    Returns:
        aggregated state dict
    """
    n_clients = len(client_states)
    if client_weights is None:
        client_weights = [1.0 / n_clients] * n_clients

    # Normalize weights
    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]

    aggregated = OrderedDict()
    for key in global_model_state.keys():
        aggregated[key] = torch.zeros_like(global_model_state[key], dtype=torch.float32)
        for client_state, weight in zip(client_states, client_weights):
            if key in client_state:
                aggregated[key] += weight * client_state[key].float()
        aggregated[key] = aggregated[key].to(global_model_state[key].dtype)

    return aggregated


def federated_prox(
    global_model_state: OrderedDict,
    client_states: List[OrderedDict],
    client_weights: Optional[List[float]] = None,
    mu: float = 0.01,
) -> OrderedDict:
    """Federated Proximal (FedProx).

    Like FedAvg but with proximal term that keeps client updates
    close to the global model, improving convergence with heterogeneous data.

    Args:
        mu: proximal term coefficient
    """
    aggregated = federated_averaging(global_model_state, client_states, client_weights)

    # Apply proximal regularization toward global model
    for key in aggregated.keys():
        if aggregated[key].dtype.is_floating_point:
            diff = aggregated[key] - global_model_state[key].float()
            aggregated[key] = (
                global_model_state[key].float() + diff / (1.0 + mu)
            ).to(global_model_state[key].dtype)

    return aggregated


def federated_scaffold(
    global_model_state: OrderedDict,
    client_states: List[OrderedDict],
    client_controls: List[OrderedDict],
    global_control: OrderedDict,
    client_weights: Optional[List[float]] = None,
    learning_rate: float = 1.0,
) -> Tuple[OrderedDict, OrderedDict]:
    """SCAFFOLD: Stochastic Controlled Averaging.

    Uses control variates to reduce client drift caused by data heterogeneity.

    Args:
        client_controls: per-client control variates
        global_control: global control variate
    Returns:
        (aggregated_state, updated_global_control)
    """
    n_clients = len(client_states)
    if client_weights is None:
        client_weights = [1.0 / n_clients] * n_clients

    total_weight = sum(client_weights)
    client_weights = [w / total_weight for w in client_weights]

    aggregated = OrderedDict()
    new_global_control = OrderedDict()

    for key in global_model_state.keys():
        # Aggregate model updates
        delta_sum = torch.zeros_like(global_model_state[key], dtype=torch.float32)
        for client_state, weight in zip(client_states, client_weights):
            if key in client_state:
                delta = client_state[key].float() - global_model_state[key].float()
                delta_sum += weight * delta

        aggregated[key] = (
            global_model_state[key].float() + learning_rate * delta_sum
        ).to(global_model_state[key].dtype)

        # Update global control variate
        if key in global_control:
            control_delta = torch.zeros_like(global_control[key], dtype=torch.float32)
            for client_ctrl, weight in zip(client_controls, client_weights):
                if key in client_ctrl:
                    control_delta += weight * (
                        client_ctrl[key].float() - global_control[key].float()
                    )
            new_global_control[key] = (
                global_control[key].float() + control_delta
            ).to(global_control[key].dtype)
        else:
            new_global_control[key] = global_model_state[key]

    return aggregated, new_global_control


def quality_weighted_aggregation(
    global_model_state: OrderedDict,
    client_states: List[OrderedDict],
    client_metrics: List[Dict[str, float]],
    quality_metric: str = "loss",
    inverse: bool = True,
) -> OrderedDict:
    """Quality-weighted aggregation based on client performance metrics.

    Weights clients by their data quality / model performance.

    Args:
        client_metrics: list of dict with metric values per client
        quality_metric: which metric to use for weighting
        inverse: if True, lower metric = higher weight (for loss)
    """
    weights = []
    for metrics in client_metrics:
        val = metrics.get(quality_metric, 1.0)
        if inverse:
            weights.append(1.0 / (val + 1e-8))
        else:
            weights.append(val)

    return federated_averaging(global_model_state, client_states, weights)
