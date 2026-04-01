"""
Federated Learning API Server.

Provides REST endpoints for federated training coordination:
- POST /fl/register: Register new client
- POST /fl/get_model: Download current global model
- POST /fl/submit_update: Submit client model update
- GET /fl/status: Training status
"""

import torch
import json
import io
import base64
import logging
from typing import Optional
from collections import OrderedDict
from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)


def create_federated_api(
    global_model: torch.nn.Module,
    aggregation_strategy: str = "fedavg",
    host: str = "0.0.0.0",
    port: int = 5001,
) -> Flask:
    """Create Flask server for federated learning coordination."""

    app = Flask(__name__)

    state = {
        "round": 0,
        "registered_clients": {},
        "pending_updates": [],
        "min_clients_per_round": 2,
    }

    @app.route("/fl/register", methods=["POST"])
    def register_client():
        data = request.get_json()
        client_id = data.get("client_id")
        if not client_id:
            return jsonify({"error": "client_id required"}), 400

        state["registered_clients"][client_id] = {
            "chemistry": data.get("chemistry", "unknown"),
            "n_samples": data.get("n_samples", 0),
            "status": "registered",
        }

        return jsonify({
            "status": "registered",
            "client_id": client_id,
            "current_round": state["round"],
        })

    @app.route("/fl/get_model", methods=["GET"])
    def get_model():
        """Return current global model weights."""
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        buffer.seek(0)

        model_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({
            "round": state["round"],
            "model_state_b64": model_b64,
        })

    @app.route("/fl/submit_update", methods=["POST"])
    def submit_update():
        """Receive client model update."""
        data = request.get_json()
        client_id = data.get("client_id")
        if not client_id:
            return jsonify({"error": "client_id required"}), 400

        model_b64 = data.get("model_state_b64")
        if not model_b64:
            return jsonify({"error": "model_state_b64 required"}), 400

        # Decode model state
        buffer = io.BytesIO(base64.b64decode(model_b64))
        client_state = torch.load(buffer, map_location="cpu", weights_only=True)

        state["pending_updates"].append({
            "client_id": client_id,
            "model_state": client_state,
            "n_samples": data.get("n_samples", 1),
            "train_loss": data.get("train_loss", None),
        })

        # Check if we have enough updates to aggregate
        if len(state["pending_updates"]) >= state["min_clients_per_round"]:
            _aggregate_updates(global_model, state)

        return jsonify({
            "status": "accepted",
            "pending_updates": len(state["pending_updates"]),
            "current_round": state["round"],
        })

    @app.route("/fl/status", methods=["GET"])
    def fl_status():
        return jsonify({
            "round": state["round"],
            "registered_clients": len(state["registered_clients"]),
            "pending_updates": len(state["pending_updates"]),
            "aggregation_strategy": aggregation_strategy,
            "clients": {
                k: {"chemistry": v["chemistry"], "status": v["status"]}
                for k, v in state["registered_clients"].items()
            },
        })

    def _aggregate_updates(model, state):
        """Aggregate pending updates into global model."""
        from mambafededge.federated.aggregation import federated_averaging

        global_state = model.state_dict()
        client_states = [u["model_state"] for u in state["pending_updates"]]
        client_weights = [float(u["n_samples"]) for u in state["pending_updates"]]

        aggregated = federated_averaging(global_state, client_states, client_weights)
        model.load_state_dict(aggregated)

        state["round"] += 1
        state["pending_updates"] = []
        logger.info(f"Aggregated updates. Round: {state['round']}")

    return app
