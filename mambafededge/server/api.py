"""
Edge Inference API Server.

Flask-based REST API for edge BMS inference:
- POST /predict: Single sensor reading prediction
- POST /predict_batch: Batch predictions
- GET /health: Server and model health status
- GET /model/info: Model metadata and statistics
- POST /model/update: Federated model update endpoint
"""

import torch
import json
import time
import logging
from typing import Optional
from flask import Flask, request, jsonify

from mambafededge.edge.runtime import EdgeRuntime

logger = logging.getLogger(__name__)


def create_edge_server(
    runtime: EdgeRuntime,
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
) -> Flask:
    """Create Flask edge inference server.

    Args:
        runtime: EdgeRuntime instance with loaded model
        host: bind address
        port: bind port
        debug: Flask debug mode
    Returns:
        Flask app
    """
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        stats = runtime.get_stats()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "device": str(runtime.device),
            "inference_stats": stats,
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        required = ["current", "voltage", "temperature"]
        if not all(k in data for k in required):
            return jsonify({"error": f"Missing fields. Required: {required}"}), 400

        # Validate input ranges
        if not (-100 <= data["current"] <= 100):
            return jsonify({"error": "Current out of range [-100, 100]A"}), 400
        if not (0 <= data["voltage"] <= 10):
            return jsonify({"error": "Voltage out of range [0, 10]V"}), 400

        result = runtime.infer(data)
        return jsonify(result)

    @app.route("/predict_batch", methods=["POST"])
    def predict_batch():
        data = request.get_json()
        if not data or "readings" not in data:
            return jsonify({"error": "Expected {'readings': [...]}"}), 400

        readings = data["readings"]
        if not isinstance(readings, list) or len(readings) > 1000:
            return jsonify({"error": "readings must be a list with max 1000 items"}), 400

        results = runtime.infer_batch(readings)
        return jsonify({"predictions": results})

    @app.route("/model/info", methods=["GET"])
    def model_info():
        model = runtime.model
        total_params = sum(p.numel() for p in model.parameters())
        return jsonify({
            "model_type": type(model).__name__,
            "total_parameters": total_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "device": str(runtime.device),
            "buffer_size": runtime.buffer_size,
            "max_latency_ms": runtime.max_latency_ms,
        })

    @app.route("/model/reset", methods=["POST"])
    def reset():
        runtime.reset()
        return jsonify({"status": "reset complete"})

    @app.route("/stats", methods=["GET"])
    def stats():
        return jsonify(runtime.get_stats())

    return app
