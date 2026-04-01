"""
Federated Learning Client.

Represents an edge device (EV, energy storage system) that:
- Trains locally on private battery data
- Sends only model updates (not raw data)
- Supports differential privacy
- Manages local data and model checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any
from collections import OrderedDict
import copy


class FederatedClient:
    """Federated learning client for edge BMS devices."""

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        local_epochs: int = 5,
        optimizer_cls: type = optim.Adam,
        loss_fn: Optional[Callable] = None,
        device: str = "cpu",
        chemistry_type: str = "unknown",
        dp_epsilon: Optional[float] = None,
        dp_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.device = device
        self.chemistry_type = chemistry_type
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.max_grad_norm = max_grad_norm

        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)

        if loss_fn is None:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = loss_fn

        # SCAFFOLD control variate
        self._control_variate = None
        self._global_control = None

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "rounds": [],
        }

    def get_model_state(self) -> OrderedDict:
        """Get current model state dict."""
        return copy.deepcopy(self.model.state_dict())

    def set_model_state(self, state_dict: OrderedDict):
        """Update model from global/aggregated state."""
        self.model.load_state_dict(state_dict)

    def get_control_variate(self) -> Optional[OrderedDict]:
        return self._control_variate

    def set_global_control(self, control: OrderedDict):
        self._global_control = control

    def local_train(
        self,
        global_state: Optional[OrderedDict] = None,
        mu: float = 0.0,  # FedProx proximal term
    ) -> Dict[str, Any]:
        """Perform local training rounds.

        Args:
            global_state: global model state for FedProx regularization
            mu: FedProx proximal coefficient

        Returns:
            dict with updated state, metrics, data_size
        """
        if global_state is not None:
            self.set_model_state(global_state)

        # Save initial state for SCAFFOLD
        initial_state = self.get_model_state()

        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch in self.train_loader:
                self.optimizer.zero_grad()

                # Unpack batch
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                elif isinstance(batch, dict):
                    inputs = batch["input"].to(self.device)
                    targets = batch["target"].to(self.device)
                else:
                    raise ValueError(f"Unexpected batch type: {type(batch)}")

                # Forward pass
                outputs = self.model(inputs)
                if isinstance(outputs, dict):
                    pred = outputs.get("soh", outputs.get("mean", list(outputs.values())[0]))
                elif isinstance(outputs, tuple):
                    pred = outputs[0]
                else:
                    pred = outputs

                # Ensure shape compatibility
                if pred.shape != targets.shape:
                    if pred.dim() > targets.dim():
                        pred = pred.squeeze(-1)
                    if pred.dim() == 3 and targets.dim() == 2:
                        pred = pred[:, -1, :]

                loss = self.loss_fn(pred, targets)

                # FedProx regularization
                if mu > 0 and global_state is not None:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_state:
                            prox_loss += ((param - global_state[name].to(self.device)) ** 2).sum()
                    loss = loss + (mu / 2.0) * prox_loss

                loss.backward()

                # Gradient clipping (for DP and stability)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

                # Add DP noise if enabled
                if self.dp_epsilon is not None:
                    self._add_dp_noise()

                self.optimizer.step()

                epoch_loss += loss.item() * inputs.shape[0]
                n_samples += inputs.shape[0]

            total_loss += epoch_loss

        avg_loss = total_loss / (n_samples * self.local_epochs + 1e-8)

        # Update SCAFFOLD control variate
        if self._global_control is not None:
            self._update_control_variate(initial_state)

        # Validation
        val_loss = self._validate() if self.val_loader is not None else None

        self.history["train_loss"].append(avg_loss)
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)

        return {
            "client_id": self.client_id,
            "model_state": self.get_model_state(),
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "n_samples": n_samples,
            "chemistry_type": self.chemistry_type,
            "control_variate": self._control_variate,
        }

    def _validate(self) -> float:
        """Run validation loop."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

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
                if isinstance(outputs, dict):
                    pred = outputs.get("soh", outputs.get("mean", list(outputs.values())[0]))
                elif isinstance(outputs, tuple):
                    pred = outputs[0]
                else:
                    pred = outputs

                if pred.shape != targets.shape:
                    if pred.dim() > targets.dim():
                        pred = pred.squeeze(-1)
                    if pred.dim() == 3 and targets.dim() == 2:
                        pred = pred[:, -1, :]

                loss = self.loss_fn(pred, targets)
                total_loss += loss.item() * inputs.shape[0]
                n_samples += inputs.shape[0]

        return total_loss / (n_samples + 1e-8)

    def _add_dp_noise(self):
        """Add calibrated Gaussian noise for differential privacy."""
        if self.dp_epsilon is None:
            return

        sensitivity = self.max_grad_norm
        sigma = sensitivity * (2 * torch.log(torch.tensor(1.25 / self.dp_delta))).sqrt() / self.dp_epsilon

        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * sigma
                param.grad.add_(noise)

    def _update_control_variate(self, initial_state: OrderedDict):
        """Update SCAFFOLD control variate."""
        if self._global_control is None:
            return

        new_control = OrderedDict()
        current_state = self.get_model_state()

        for key in current_state.keys():
            if key in self._global_control:
                if current_state[key].dtype.is_floating_point:
                    c_new = (
                        self._global_control[key].float()
                        - (current_state[key].float() - initial_state[key].float())
                        / (self.local_epochs * self.lr + 1e-8)
                    )
                    new_control[key] = c_new.to(current_state[key].dtype)
                else:
                    new_control[key] = current_state[key]

        self._control_variate = new_control
