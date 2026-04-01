"""Edge Server for MambaFedEdge."""

from mambafededge.server.api import create_edge_server
from mambafededge.server.federated_server import create_federated_api

__all__ = [
    "create_edge_server",
    "create_federated_api",
]
