"""
Edge Runtime for on-device inference.

Provides a lightweight runtime for BMS microcontrollers:
- Streaming inference (one sample at a time)
- State management across time steps
- Watchdog timer integration
- Memory-efficient buffering
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class EdgeRuntime:
    """Lightweight runtime for edge device BMS inference.

    Manages:
    - Model state across streaming inference
    - Watchdog monitoring
    - Input buffering and preprocessing
    - Output caching and smoothing
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        max_latency_ms: float = 25.0,
        buffer_size: int = 100,
        smoothing_window: int = 5,
        watchdog_enabled: bool = True,
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.max_latency_ms = max_latency_ms
        self.buffer_size = buffer_size
        self.smoothing_window = smoothing_window
        self.watchdog_enabled = watchdog_enabled

        # State management
        self._model_state: Optional[Dict] = None
        self._input_buffer: list = []
        self._output_history: list = []
        self._step_count: int = 0

        # Performance tracking
        self._latencies: list = []
        self._faults: int = 0

        # Move model to device
        self.model = self.model.to(device)

    def reset(self):
        """Reset runtime state."""
        self._model_state = None
        self._input_buffer = []
        self._output_history = []
        self._step_count = 0
        self._latencies = []
        self._faults = 0

    def preprocess(self, raw_input: Dict[str, float]) -> torch.Tensor:
        """Preprocess raw sensor readings.

        Args:
            raw_input: dict with keys: current, voltage, temperature, time
        Returns:
            (1, d_input) tensor
        """
        values = [
            raw_input.get("current", 0.0),
            raw_input.get("voltage", 3.7),
            raw_input.get("temperature", 298.0),
            raw_input.get("time", 0.0),
        ]
        return torch.tensor([values], dtype=torch.float32, device=self.device)

    def infer(
        self,
        sensor_reading: Dict[str, float],
    ) -> Dict[str, Any]:
        """Run single-step inference.

        Args:
            sensor_reading: dict with current, voltage, temperature, time
        Returns:
            dict with predictions, uncertainty, latency, safety status
        """
        start_time = time.perf_counter()

        # Preprocess
        x = self.preprocess(sensor_reading)

        # Buffer management
        self._input_buffer.append(x)
        if len(self._input_buffer) > self.buffer_size:
            self._input_buffer = self._input_buffer[-self.buffer_size:]

        # Build sequence from buffer
        x_seq = torch.cat(self._input_buffer, dim=0).unsqueeze(0)  # (1, L, D)

        # Inference
        with torch.no_grad():
            try:
                output = self.model(x_seq)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                self._faults += 1
                return self._fallback_response(str(e))

        # Measure latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._latencies.append(latency_ms)

        # Process output
        result = self._process_output(output, latency_ms)
        self._output_history.append(result)
        self._step_count += 1

        # Apply temporal smoothing
        if len(self._output_history) >= self.smoothing_window:
            result = self._smooth_output(result)

        # Watchdog check
        if self.watchdog_enabled:
            result["watchdog"] = self._watchdog_check(result, latency_ms)

        return result

    def infer_batch(
        self,
        sensor_readings: list,
    ) -> list:
        """Run batch inference.

        Args:
            sensor_readings: list of sensor reading dicts
        Returns:
            list of prediction dicts
        """
        results = []
        for reading in sensor_readings:
            results.append(self.infer(reading))
        return results

    def _process_output(self, output: Any, latency_ms: float) -> Dict[str, Any]:
        """Convert model output to structured result."""
        result = {"latency_ms": latency_ms, "step": self._step_count}

        if isinstance(output, dict):
            for key, val in output.items():
                if isinstance(val, torch.Tensor):
                    result[key] = val.cpu().item() if val.numel() == 1 else val.cpu().tolist()
                elif isinstance(val, dict):
                    result[key] = {
                        k: v.cpu().item() if isinstance(v, torch.Tensor) and v.numel() == 1
                        else v.cpu().tolist() if isinstance(v, torch.Tensor) else v
                        for k, v in val.items()
                    }
        elif isinstance(output, tuple):
            pred = output[0]
            result["prediction"] = pred.cpu().item() if pred.numel() == 1 else pred.cpu().tolist()
            if len(output) > 1:
                result["state"] = output[1]
        elif isinstance(output, torch.Tensor):
            result["prediction"] = output.cpu().item() if output.numel() == 1 else output.cpu().tolist()

        return result

    def _smooth_output(self, current: Dict[str, Any]) -> Dict[str, Any]:
        """Apply exponential moving average smoothing."""
        alpha = 2.0 / (self.smoothing_window + 1)
        recent = self._output_history[-self.smoothing_window:]

        for key in ["soh", "rul", "prediction"]:
            if key in current and isinstance(current[key], (int, float)):
                vals = [r.get(key, current[key]) for r in recent if isinstance(r.get(key), (int, float))]
                if vals:
                    ema = vals[0]
                    for v in vals[1:]:
                        ema = alpha * v + (1 - alpha) * ema
                    current[f"{key}_smoothed"] = ema

        return current

    def _watchdog_check(self, result: Dict, latency_ms: float) -> Dict[str, bool]:
        """Safety watchdog checks."""
        checks = {
            "latency_ok": latency_ms < self.max_latency_ms,
            "no_nan": all(
                not (isinstance(v, float) and (v != v))
                for v in result.values()
                if isinstance(v, (int, float))
            ),
            "fault_count": self._faults,
            "healthy": True,
        }

        soh = result.get("soh")
        if isinstance(soh, (int, float)):
            checks["soh_valid"] = 0.0 <= soh <= 1.0
        else:
            checks["soh_valid"] = True

        checks["healthy"] = all(
            v for k, v in checks.items()
            if k not in ("fault_count",) and isinstance(v, bool)
        )

        return checks

    def _fallback_response(self, error: str) -> Dict[str, Any]:
        """Generate fallback response on error."""
        return {
            "error": error,
            "fallback": True,
            "soh": 0.5,  # Conservative estimate
            "confidence": 0.0,
            "step": self._step_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime performance statistics."""
        if not self._latencies:
            return {"status": "no inferences performed"}

        lat = self._latencies
        return {
            "total_inferences": self._step_count,
            "mean_latency_ms": sum(lat) / len(lat),
            "min_latency_ms": min(lat),
            "max_latency_ms": max(lat),
            "p95_latency_ms": sorted(lat)[int(0.95 * len(lat))] if len(lat) > 1 else lat[0],
            "faults": self._faults,
            "buffer_size": len(self._input_buffer),
            "latency_violations": sum(1 for l in lat if l > self.max_latency_ms),
        }
