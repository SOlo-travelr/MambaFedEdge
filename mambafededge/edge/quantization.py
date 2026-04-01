"""
Edge-Aware Quantization for BMS Deployment.

Compresses models for resource-constrained edge devices:
- Post-training quantization (PTQ): float32 -> int8
- Quantization-aware training (QAT)
- Weight pruning
- Knowledge distillation support
- ONNX export for MCU deployment
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
import copy
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    method: str = "dynamic"           # "dynamic", "static", "qat"
    dtype: str = "qint8"              # "qint8", "float16"
    calibration_batches: int = 100
    prune_ratio: float = 0.0         # 0.0 = no pruning, 0.5 = remove 50% weights
    prune_method: str = "magnitude"  # "magnitude", "structured"
    export_onnx: bool = False
    onnx_opset: int = 13
    target_size_mb: Optional[float] = None  # Target model size


class ModelQuantizer:
    """Quantizes PyTorch models for edge deployment."""

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Quantize model based on config.

        Args:
            model: PyTorch model to quantize
            calibration_data: data for static quantization calibration
        Returns:
            quantized model
        """
        if self.config.method == "dynamic":
            return self._dynamic_quantize(model)
        elif self.config.method == "static":
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration data")
            return self._static_quantize(model, calibration_data)
        elif self.config.method == "qat":
            return self._prepare_qat(model)
        else:
            raise ValueError(f"Unknown quantization method: {self.config.method}")

    def _dynamic_quantize(self, model: nn.Module) -> nn.Module:
        """Dynamic quantization - quantizes weights, activations at runtime."""
        model_copy = copy.deepcopy(model)
        model_copy.eval()

        quantized = torch.quantization.quantize_dynamic(
            model_copy,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8,
        )

        logger.info("Applied dynamic quantization (int8)")
        return quantized

    def _static_quantize(
        self, model: nn.Module, calibration_data: torch.Tensor
    ) -> nn.Module:
        """Static quantization with calibration data."""
        model_copy = copy.deepcopy(model)
        model_copy.eval()

        # Fuse operations where possible
        model_copy = self._fuse_modules(model_copy)

        # Set quantization config
        model_copy.qconfig = torch.quantization.get_default_qconfig("x86")

        # Prepare and calibrate
        prepared = torch.quantization.prepare(model_copy)

        with torch.no_grad():
            if calibration_data.dim() == 2:
                calibration_data = calibration_data.unsqueeze(0)
            for i in range(min(self.config.calibration_batches, calibration_data.shape[0])):
                prepared(calibration_data[i:i+1])

        quantized = torch.quantization.convert(prepared)
        logger.info("Applied static quantization with calibration")
        return quantized

    def _prepare_qat(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization-aware training."""
        model_copy = copy.deepcopy(model)
        model_copy.train()

        model_copy.qconfig = torch.quantization.get_default_qat_qconfig("x86")
        prepared = torch.quantization.prepare_qat(model_copy)

        logger.info("Prepared model for quantization-aware training")
        return prepared

    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse Conv-BN-ReLU and Linear-ReLU sequences."""
        # This is a simplified version - production would need model-specific fusion
        return model

    def prune(self, model: nn.Module) -> nn.Module:
        """Apply weight pruning to reduce model size."""
        if self.config.prune_ratio <= 0:
            return model

        model_copy = copy.deepcopy(model)
        import torch.nn.utils.prune as prune

        # Apply unstructured pruning to all linear layers
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Linear):
                if self.config.prune_method == "magnitude":
                    prune.l1_unstructured(module, name="weight", amount=self.config.prune_ratio)
                elif self.config.prune_method == "structured":
                    prune.ln_structured(module, name="weight", amount=self.config.prune_ratio, n=2, dim=0)

        # Make pruning permanent
        for name, module in model_copy.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, "weight")
                except ValueError:
                    pass

        n_zeros = sum((p == 0).sum().item() for p in model_copy.parameters())
        n_total = sum(p.numel() for p in model_copy.parameters())
        logger.info(f"Pruned {n_zeros}/{n_total} parameters ({100*n_zeros/n_total:.1f}% sparsity)")

        return model_copy

    def export_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        input_names: list = None,
        output_names: list = None,
    ):
        """Export model to ONNX format for MCU deployment."""
        model.eval()
        input_names = input_names or ["sensor_input"]
        output_names = output_names or ["soh_prediction"]

        torch.onnx.export(
            model,
            sample_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=self.config.onnx_opset,
            dynamic_axes={
                input_names[0]: {0: "batch", 1: "sequence"},
                output_names[0]: {0: "batch"},
            },
        )
        logger.info(f"Exported ONNX model to {output_path}")

    def get_model_stats(self, model: nn.Module) -> Dict[str, Any]:
        """Get model size and performance statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate sizes
        fp32_size = total_params * 4 / (1024 * 1024)
        fp16_size = total_params * 2 / (1024 * 1024)
        int8_size = total_params * 1 / (1024 * 1024)

        # Count sparse (zero) parameters
        n_zeros = sum((p == 0).sum().item() for p in model.parameters())
        sparsity = n_zeros / total_params if total_params > 0 else 0

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "fp32_size_mb": fp32_size,
            "fp16_size_mb": fp16_size,
            "int8_size_mb": int8_size,
            "sparsity": sparsity,
            "effective_size_mb": int8_size * (1 - sparsity),
        }

    def benchmark_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        n_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Benchmark inference latency."""
        import time

        model.eval()
        times = []

        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                _ = model(sample_input)

            # Benchmark
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(sample_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms

        return {
            "mean_latency_ms": sum(times) / len(times),
            "min_latency_ms": min(times),
            "max_latency_ms": max(times),
            "p95_latency_ms": sorted(times)[int(0.95 * len(times))],
            "p99_latency_ms": sorted(times)[int(0.99 * len(times))],
            "throughput_per_sec": 1000.0 / (sum(times) / len(times)),
        }
