"""Configuration management."""

import yaml
import json
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    d_sensor: int = 4
    d_model: int = 128
    d_state: int = 64
    encoder_layers: int = 6
    decoder_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    uncertainty_method: str = "evidential"


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    batch_size: int = 32
    seq_len: int = 100
    early_stopping_patience: int = 15
    gradient_clip: float = 1.0


@dataclass
class FederatedConfig:
    n_rounds: int = 100
    clients_per_round: int = 5
    local_epochs: int = 5
    aggregation_strategy: str = "fedavg"
    mu: float = 0.01
    dp_epsilon: Optional[float] = None


@dataclass
class EdgeConfig:
    quantize: bool = True
    prune: bool = False
    prune_ratio: float = 0.3
    max_latency_ms: float = 25.0
    buffer_size: int = 100


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        config = cls()
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "federated" in data:
            config.federated = FederatedConfig(**data["federated"])
        if "edge" in data:
            config.edge = EdgeConfig(**data["edge"])
        for key in ["data_dir", "checkpoint_dir", "device"]:
            if key in data:
                setattr(config, key, data[key])
        return config

    def to_dict(self) -> Dict:
        return asdict(self)
