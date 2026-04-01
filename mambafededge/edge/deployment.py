"""
Edge Deployment Utilities.

Handles:
- Model packaging for edge devices
- ONNX/TFLite export
- Deployment configuration
- Edge server communication
"""

import torch
import torch.nn as nn
import os
import json
import logging
from typing import Dict, Optional, Any
from mambafededge.edge.quantization import ModelQuantizer, QuantizationConfig

logger = logging.getLogger(__name__)


class EdgeDeployer:
    """Deploy models to edge devices."""

    def __init__(self, output_dir: str = "deployments"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def package_model(
        self,
        model: nn.Module,
        model_name: str,
        quantize: bool = True,
        prune: bool = False,
        sample_input: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Package model for edge deployment.

        Args:
            model: trained PyTorch model
            model_name: name for the deployment package
            quantize: apply quantization
            prune: apply weight pruning
            sample_input: sample input for ONNX export/calibration
        Returns:
            path to deployment package
        """
        pkg_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(pkg_dir, exist_ok=True)

        # Save original model
        original_path = os.path.join(pkg_dir, "model_fp32.pt")
        torch.save(model.state_dict(), original_path)
        logger.info(f"Saved FP32 model: {original_path}")

        quantizer = ModelQuantizer(QuantizationConfig())

        # Get original stats
        original_stats = quantizer.get_model_stats(model)

        # Quantize
        if quantize:
            quant_model = quantizer.quantize(model)
            quant_path = os.path.join(pkg_dir, "model_int8.pt")
            torch.save(quant_model.state_dict(), quant_path)
            logger.info(f"Saved INT8 model: {quant_path}")

        # Prune
        if prune:
            quantizer.config.prune_ratio = 0.3
            pruned_model = quantizer.prune(model)
            pruned_path = os.path.join(pkg_dir, "model_pruned.pt")
            torch.save(pruned_model.state_dict(), pruned_path)

        # Export ONNX
        if sample_input is not None:
            onnx_path = os.path.join(pkg_dir, "model.onnx")
            try:
                quantizer.export_onnx(model, sample_input, onnx_path)
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")

        # Benchmark
        if sample_input is not None:
            perf = quantizer.benchmark_inference(model, sample_input)
        else:
            perf = {}

        # Save deployment manifest
        manifest = {
            "model_name": model_name,
            "original_stats": original_stats,
            "performance": perf,
            "quantized": quantize,
            "pruned": prune,
            "metadata": metadata or {},
            "files": os.listdir(pkg_dir),
        }

        manifest_path = os.path.join(pkg_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(f"Deployment package: {pkg_dir}")
        return pkg_dir

    def load_deployed_model(
        self,
        model: nn.Module,
        pkg_dir: str,
        variant: str = "model_fp32.pt",
    ) -> nn.Module:
        """Load a deployed model variant."""
        path = os.path.join(pkg_dir, variant)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        return model
