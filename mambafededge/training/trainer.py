"""
Training Engine for MambaFedEdge models.

Supports:
- Centralized training
- Federated training coordination
- Physics-informed losses
- Mixed precision training
- Learning rate scheduling
- Checkpointing and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict, List, Callable, Any
import time
import os
import logging
import json

from mambafededge.training.losses import PhysicsInformedLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Training engine for battery health prediction models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 10,
        early_stopping_patience: int = 15,
        gradient_clip: float = 1.0,
        use_amp: bool = False,
    ):
        # Device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.log_interval = log_interval
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp and self.device == "cuda"

        # Loss
        self.loss_fn = loss_fn or PhysicsInformedLoss()

        # Optimizer
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = scheduler or optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )

        # AMP scaler
        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "epoch_time": [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            elif isinstance(batch, dict):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
            else:
                continue

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device):
                    outputs = self.model(inputs)
                    loss_dict = self._compute_loss(outputs, targets)
                    loss = loss_dict["total"]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss_dict = self._compute_loss(outputs, targets)
                loss = loss_dict["total"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.optimizer.step()

            total_loss += loss.item()
            total_mse += loss_dict.get("mse", loss).item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_mse = total_mse / max(n_batches, 1)

        return {"loss": avg_loss, "mse": avg_mse}

    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    inputs = batch["input"].to(self.device)
                    targets = batch["target"].to(self.device)
                else:
                    continue

                outputs = self.model(inputs)
                loss_dict = self._compute_loss(outputs, targets)

                total_loss += loss_dict["total"].item()
                total_mse += loss_dict.get("mse", loss_dict["total"]).item()

                # Collect predictions for metrics
                pred = self._extract_prediction(outputs)
                all_preds.append(pred.cpu())
                all_targets.append(targets.cpu())
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_mse = total_mse / max(n_batches, 1)

        # Global metrics
        all_preds = torch.cat(all_preds).squeeze()
        all_targets = torch.cat(all_targets).squeeze()
        mae = (all_preds - all_targets).abs().mean().item()
        rmse = ((all_preds - all_targets) ** 2).mean().sqrt().item()

        # R² score
        ss_res = ((all_targets - all_preds) ** 2).sum()
        ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            "loss": avg_loss,
            "mse": avg_mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2.item(),
        }

    def train(
        self,
        n_epochs: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, List]:
        """Full training loop.

        Returns: training history dict
        """
        n_epochs = n_epochs or self.max_epochs
        logger.info(f"Starting training: {n_epochs} epochs on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, n_epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - start_time

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics.get("loss"))
            self.history["lr"].append(lr)
            self.history["epoch_time"].append(epoch_time)

            if epoch % self.log_interval == 0 or epoch == 1:
                log_msg = (
                    f"Epoch {epoch:03d}/{n_epochs} | "
                    f"Train: {train_metrics['loss']:.6f} | "
                )
                if val_metrics:
                    log_msg += (
                        f"Val: {val_metrics['loss']:.6f} | "
                        f"MAE: {val_metrics.get('mae', 0):.6f} | "
                        f"R²: {val_metrics.get('r2', 0):.4f} | "
                    )
                log_msg += f"LR: {lr:.2e} | Time: {epoch_time:.1f}s"
                logger.info(log_msg)
                print(log_msg)

            # Early stopping
            val_loss = val_metrics.get("loss", train_metrics["loss"])
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.checkpoint_dir:
                    self._save_checkpoint(epoch, "best")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                print(f"Early stopping at epoch {epoch}")
                break

            if callback:
                callback(epoch, train_metrics, val_metrics)

        # Save final
        if self.checkpoint_dir:
            self._save_checkpoint(epoch, "final")

        return self.history

    def _compute_loss(
        self, outputs: Any, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute loss from model outputs."""
        # Extract prediction tensor
        pred = self._extract_prediction(outputs)

        # Ensure shape compatibility
        if pred.shape != targets.shape:
            if pred.dim() > targets.dim():
                pred = pred.squeeze(-1)
            if pred.dim() == 3 and targets.dim() == 1:
                pred = pred[:, -1, 0] if pred.shape[-1] == 1 else pred[:, -1, :]

        if isinstance(self.loss_fn, PhysicsInformedLoss):
            return self.loss_fn(pred, targets)
        else:
            loss = self.loss_fn(pred, targets)
            return {"total": loss, "mse": loss}

    def _extract_prediction(self, outputs: Any) -> torch.Tensor:
        """Extract prediction tensor from model output."""
        if isinstance(outputs, dict):
            for key in ["soh", "mean", "prediction"]:
                if key in outputs:
                    return outputs[key]
            # Return first tensor value
            for v in outputs.values():
                if isinstance(v, torch.Tensor):
                    return v
        elif isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def _save_checkpoint(self, epoch: int, tag: str):
        """Save training checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.history = ckpt.get("history", self.history)
        logger.info(f"Loaded checkpoint from {path}")

    def get_summary(self) -> Dict:
        """Get training summary."""
        return {
            "device": str(self.device),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.history["train_loss"]),
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
        }
