"""
Battery Dataset Utilities.

- Generic BatteryDataset for loading battery cycling data
- SyntheticBatteryGenerator for creating realistic synthetic data
  for testing and development
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Dict, List
import math


class BatteryDataset(Dataset):
    """Generic battery cycling dataset.

    Supports loading from:
    - Numpy arrays
    - Dict of arrays
    - Pre-processed tensors
    """

    def __init__(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        seq_len: int = 100,
        stride: int = 1,
        chemistry: str = "unknown",
        normalize: bool = True,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.chemistry = chemistry

        if normalize:
            self.data_mean = data.mean(dim=0)
            self.data_std = data.std(dim=0) + 1e-8
            self.data = (data - self.data_mean) / self.data_std
        else:
            self.data = data
            self.data_mean = torch.zeros(data.shape[-1])
            self.data_std = torch.ones(data.shape[-1])

        self.targets = targets

        # Compute valid indices
        self.indices = list(range(0, len(data) - seq_len + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.seq_len
        x = self.data[start:end]
        y = self.targets[min(end - 1, len(self.targets) - 1)]
        return x, y

    def get_stats(self) -> Dict:
        return {
            "n_samples": len(self),
            "seq_len": self.seq_len,
            "n_features": self.data.shape[-1],
            "chemistry": self.chemistry,
            "data_mean": self.data_mean.tolist(),
            "data_std": self.data_std.tolist(),
        }


class SyntheticBatteryGenerator:
    """Generate realistic synthetic battery cycling data.

    Physics-based simulation including:
    - Voltage curves (OCV + ohmic + polarization)
    - Current profiles (CC-CV charging, discharge)
    - Temperature dynamics (heat generation + cooling)
    - Degradation (SEI growth, Li plating, capacity fade)
    """

    def __init__(
        self,
        chemistry: str = "NMC111",
        nominal_capacity: float = 3.0,  # Ah
        nominal_voltage: float = 3.7,   # V
        internal_resistance: float = 0.03,  # Ohm
        thermal_mass: float = 100.0,     # J/K
        cooling_coeff: float = 5.0,      # W/K
        ambient_temp: float = 298.0,     # K
        seed: Optional[int] = None,
    ):
        self.chemistry = chemistry
        self.Q_nom = nominal_capacity
        self.V_nom = nominal_voltage
        self.R_int = internal_resistance
        self.thermal_mass = thermal_mass
        self.cooling_coeff = cooling_coeff
        self.T_amb = ambient_temp

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Chemistry-dependent parameters
        self._setup_chemistry(chemistry)

    def _setup_chemistry(self, chemistry: str):
        """Set chemistry-specific electrochemical parameters."""
        params = {
            "NMC111": {"V_max": 4.2, "V_min": 3.0, "sei_rate": 0.02, "plating_risk": 0.01},
            "NMC811": {"V_max": 4.2, "V_min": 3.0, "sei_rate": 0.03, "plating_risk": 0.02},
            "LFP":    {"V_max": 3.65, "V_min": 2.5, "sei_rate": 0.01, "plating_risk": 0.005},
            "NCA":    {"V_max": 4.2, "V_min": 3.0, "sei_rate": 0.025, "plating_risk": 0.015},
            "LCO":    {"V_max": 4.2, "V_min": 3.0, "sei_rate": 0.035, "plating_risk": 0.025},
        }
        p = params.get(chemistry, params["NMC111"])
        self.V_max = p["V_max"]
        self.V_min = p["V_min"]
        self.sei_rate = p["sei_rate"]
        self.plating_risk = p["plating_risk"]

    def ocv_curve(self, soc: np.ndarray) -> np.ndarray:
        """Open Circuit Voltage as function of SOC."""
        # Simplified OCV model (polynomial fit)
        soc_clipped = np.clip(soc, 0, 1)
        ocv = (
            self.V_min
            + (self.V_max - self.V_min) * (
                -0.3 * soc_clipped**3
                + 0.8 * soc_clipped**2
                + 0.2 * soc_clipped
                + 0.3
            )
        )
        return np.clip(ocv, self.V_min, self.V_max)

    def generate_cycle(
        self,
        cycle_num: int,
        c_rate_charge: float = 0.5,
        c_rate_discharge: float = 1.0,
        dt: float = 1.0,  # seconds
        soh: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """Generate one charge-discharge cycle.

        Returns dict with: current, voltage, temperature, time, soc
        """
        Q_actual = self.Q_nom * soh
        data = {"current": [], "voltage": [], "temperature": [], "time": [], "soc": []}

        T = self.T_amb
        soc = 0.0
        t = 0.0

        # === Charging (CC-CV) ===
        I_charge = c_rate_charge * Q_actual
        # CC phase
        while soc < 0.95 and t < 7200:
            # SOC update
            dsoc = I_charge * dt / (Q_actual * 3600)
            soc = min(soc + dsoc, 1.0)

            # Voltage
            ocv = self.ocv_curve(np.array([soc]))[0]
            V = ocv + I_charge * self.R_int * (1 + 0.1 * (1 - soh))
            V = min(V, self.V_max)

            # Temperature
            Q_gen = I_charge**2 * self.R_int
            Q_cool = self.cooling_coeff * (T - self.T_amb)
            dT = (Q_gen - Q_cool) * dt / self.thermal_mass
            T += dT

            data["current"].append(I_charge)
            data["voltage"].append(V)
            data["temperature"].append(T)
            data["time"].append(t)
            data["soc"].append(soc)
            t += dt

            if V >= self.V_max:
                break

        # CV phase
        while soc < 0.99 and t < 10800:
            V_target = self.V_max
            ocv = self.ocv_curve(np.array([soc]))[0]
            I_cv = max((V_target - ocv) / (self.R_int * (1 + 0.1 * (1 - soh))), 0.05 * Q_actual)
            I_cv = max(I_cv, 0)

            dsoc = I_cv * dt / (Q_actual * 3600)
            soc = min(soc + dsoc, 1.0)

            Q_gen = I_cv**2 * self.R_int
            Q_cool = self.cooling_coeff * (T - self.T_amb)
            dT = (Q_gen - Q_cool) * dt / self.thermal_mass
            T += dT

            data["current"].append(I_cv)
            data["voltage"].append(V_target)
            data["temperature"].append(T)
            data["time"].append(t)
            data["soc"].append(soc)
            t += dt

            if I_cv < 0.05 * Q_actual:
                break

        # Rest period
        for _ in range(60):
            data["current"].append(0.0)
            ocv = self.ocv_curve(np.array([soc]))[0]
            data["voltage"].append(ocv)
            T += (self.T_amb - T) * dt * self.cooling_coeff / self.thermal_mass
            data["temperature"].append(T)
            data["time"].append(t)
            data["soc"].append(soc)
            t += dt

        # === Discharging ===
        I_discharge = -c_rate_discharge * Q_actual
        while soc > 0.05 and t < 21600:
            dsoc = I_discharge * dt / (Q_actual * 3600)
            soc = max(soc + dsoc, 0.0)

            ocv = self.ocv_curve(np.array([soc]))[0]
            V = ocv + I_discharge * self.R_int * (1 + 0.1 * (1 - soh))
            V = max(V, self.V_min)

            Q_gen = I_discharge**2 * self.R_int
            Q_cool = self.cooling_coeff * (T - self.T_amb)
            dT = (Q_gen - Q_cool) * dt / self.thermal_mass
            T += dT

            data["current"].append(I_discharge)
            data["voltage"].append(V)
            data["temperature"].append(T)
            data["time"].append(t)
            data["soc"].append(soc)
            t += dt

            if V <= self.V_min:
                break

        # Add noise
        for key in ["current", "voltage", "temperature"]:
            arr = np.array(data[key])
            noise = np.random.normal(0, 0.001 * np.abs(arr).max(), arr.shape)
            data[key] = (arr + noise).tolist()

        return {k: np.array(v) for k, v in data.items()}

    def generate_aging_dataset(
        self,
        n_cycles: int = 500,
        c_rate_charge: float = 0.5,
        c_rate_discharge: float = 1.0,
        dt: float = 10.0,
        temperature_variation: float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate aging dataset with gradual degradation.

        Returns:
            data: (N, 4) tensor [current, voltage, temperature, time]
            soh: (N,) tensor of SOH values
        """
        all_data = []
        all_soh = []

        soh = 1.0
        total_time = 0.0

        for cycle in range(n_cycles):
            # Degradation model
            sei_loss = self.sei_rate * math.sqrt(cycle + 1) / 100.0
            cycle_loss = 0.0005 * cycle
            T_actual = self.T_amb + np.random.uniform(-temperature_variation, temperature_variation)
            temp_factor = np.exp((T_actual - 298.0) / 50.0)
            calendar_loss = 0.001 * cycle * temp_factor / 100.0

            soh = max(1.0 - sei_loss - cycle_loss - calendar_loss, 0.3)

            # Vary operating conditions slightly
            c_charge = c_rate_charge * (1.0 + np.random.uniform(-0.1, 0.1))
            c_discharge = c_rate_discharge * (1.0 + np.random.uniform(-0.1, 0.1))

            cycle_data = self.generate_cycle(
                cycle_num=cycle,
                c_rate_charge=c_charge,
                c_rate_discharge=c_discharge,
                dt=dt,
                soh=soh,
            )

            n_points = len(cycle_data["current"])
            for i in range(n_points):
                all_data.append([
                    cycle_data["current"][i],
                    cycle_data["voltage"][i],
                    cycle_data["temperature"][i],
                    total_time + cycle_data["time"][i],
                ])
                all_soh.append(soh)

            total_time += cycle_data["time"][-1]

        data_tensor = torch.tensor(np.array(all_data), dtype=torch.float32)
        soh_tensor = torch.tensor(np.array(all_soh), dtype=torch.float32)

        return data_tensor, soh_tensor

    def generate_federated_datasets(
        self,
        n_clients: int = 5,
        cycles_per_client: int = 200,
        chemistries: Optional[List[str]] = None,
        seq_len: int = 50,
    ) -> List[BatteryDataset]:
        """Generate datasets for federated learning simulation.

        Each client gets a dataset with different:
        - Chemistry type
        - Operating conditions
        - Amount of data
        """
        if chemistries is None:
            chemistries = ["NMC111", "NMC811", "LFP", "NCA", "NMC111"]

        datasets = []
        for i in range(n_clients):
            chem = chemistries[i % len(chemistries)]
            self._setup_chemistry(chem)

            # Vary conditions per client
            c_charge = 0.3 + np.random.random() * 0.7
            c_discharge = 0.5 + np.random.random() * 1.5
            n_cyc = int(cycles_per_client * (0.5 + np.random.random()))

            data, soh = self.generate_aging_dataset(
                n_cycles=n_cyc,
                c_rate_charge=c_charge,
                c_rate_discharge=c_discharge,
                dt=10.0,
            )

            dataset = BatteryDataset(
                data=data,
                targets=soh,
                seq_len=seq_len,
                chemistry=chem,
            )
            datasets.append(dataset)

        return datasets
