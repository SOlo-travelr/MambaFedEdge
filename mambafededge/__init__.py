"""MambaFedEdge: Mamba-2 Federated Learning for Edge AI Battery Management."""

__version__ = "0.1.0"

from mambafededge.models import (
    Mamba2Block,
    Mamba2Encoder,
    Mamba2Decoder,
    LargeBatteryModel,
    PhysicsLSTM,
    ChemistryEmbedding,
    UncertaintyHead,
    PredictionModule,
    FallbackSystem,
)
from mambafededge.federated import FederatedServer, FederatedClient
from mambafededge.edge import ModelQuantizer, EdgeRuntime
from mambafededge.training import Trainer, PhysicsInformedLoss
from mambafededge.data import BatteryDataset, SyntheticBatteryGenerator, NASABatteryDataset
