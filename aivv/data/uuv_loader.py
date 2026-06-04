"""
UUV Yaw Data Loader

Creates next-step forecasting windows for UUV yaw data and provides
train/test tensors compatible with ACA pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class UUVDataLoader:
    def __init__(
        self,
        file_path: str,
        window_size: int = 10,
        train_ratio: float = 0.7,
        downsample: int = 20,
        horizon: int = 1,
        batch_size: int = 32,
        use_differencing: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.file_path = str(file_path)
        self.window_size = int(window_size)
        self.train_ratio = float(train_ratio)
        self.downsample = int(downsample)
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.use_differencing = bool(use_differencing)
        self.device = device

        t_all, x_all = self._load_uuv(self.file_path, self.downsample)
        self.time_all = t_all
        self.yaw_all = x_all.squeeze(-1).astype(np.float32)

        n_total = len(x_all)
        n_train = int(n_total * self.train_ratio)
        self.n_total_raw = int(n_total)
        self.n_train_raw = int(n_train)
        self.target_index_offset = int(self.window_size + self.horizon - 1)
        min_needed = self.window_size + self.horizon + 1
        if n_train < min_needed or (n_total - n_train) < min_needed:
            raise ValueError(
                f"Not enough samples for split/window/horizon. "
                f"Need at least {min_needed} in each split. "
                f"Got train={n_train}, test={n_total-n_train}."
            )

        x_train_raw = x_all[:n_train]
        x_test_raw = x_all[n_train:]
        self.x_train_raw = x_train_raw.astype(np.float32)
        self.x_test_raw = x_test_raw.astype(np.float32)

        if self.use_differencing:
            # Delta targets: dy_t = y_t - y_{t-1}
            if len(x_train_raw) < 2 or len(x_test_raw) < 2:
                raise ValueError("Differencing requires at least 2 samples in train and test splits.")
            x_train_model = np.diff(x_train_raw, axis=0).astype(np.float32)
            x_test_model = np.diff(x_test_raw, axis=0).astype(np.float32)
        else:
            x_train_model = x_train_raw
            x_test_model = x_test_raw

        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train_model)
        x_test = self.scaler.transform(x_test_model)

        Xw_train, y_train = self._build_windows(x_train, self.window_size, self.horizon)
        Xw_test, y_test = self._build_windows(x_test, self.window_size, self.horizon)
        if self.use_differencing:
            # Absolute yaw target for each test window (for plotting/reconstruction).
            y_test_raw_abs = []
            y_test_prev_raw = []
            target_idx = self.window_size + self.horizon - 1
            n_w = len(x_test_model) - target_idx
            for i in range(n_w):
                raw_target_idx = i + target_idx + 1
                y_test_prev_raw.append(x_test_raw[raw_target_idx - 1])
                y_test_raw_abs.append(x_test_raw[raw_target_idx])
            y_test_raw = np.asarray(y_test_raw_abs, dtype=np.float32)
            y_test_prev = np.asarray(y_test_prev_raw, dtype=np.float32)
            self.test_prev_raw = torch.as_tensor(y_test_prev, dtype=torch.float32)
        else:
            _, y_test_raw = self._build_windows(x_test_raw, self.window_size, self.horizon)
            self.test_prev_raw = None

        self.train_X = torch.as_tensor(Xw_train, dtype=torch.float32)
        self.train_y = torch.as_tensor(y_train, dtype=torch.float32)
        self.test_X = torch.as_tensor(Xw_test, dtype=torch.float32)
        self.test_y = torch.as_tensor(y_test, dtype=torch.float32)
        self.test_y_raw = torch.as_tensor(y_test_raw, dtype=torch.float32)

        if self.device is not None:
            self.train_X = self.train_X.to(self.device)
            self.train_y = self.train_y.to(self.device)
            self.test_X = self.test_X.to(self.device)
            self.test_y = self.test_y.to(self.device)
            self.test_y_raw = self.test_y_raw.to(self.device)
            if self.test_prev_raw is not None:
                self.test_prev_raw = self.test_prev_raw.to(self.device)

    def get_test_raw_window(self, sample_idx: int) -> Optional[np.ndarray]:
        """Return raw yaw window for a given test sample index (shape: [window, 1])."""
        if sample_idx < 0:
            return None
        start = int(sample_idx)
        end = start + int(self.window_size)
        if end > len(self.x_test_raw):
            return None
        return self.x_test_raw[start:end].astype(np.float32)

    def build_test_fault_labels(self, fault_start_index: int, fault_end_index: int) -> np.ndarray:
        """Build explicit binary labels for test windows from global target indices."""
        start_index = int(fault_start_index)
        end_index = int(fault_end_index)
        if end_index < start_index:
            raise ValueError("fault_end_index must be >= fault_start_index")

        target_indices = self.get_test_target_global_indices()
        labels = ((target_indices >= start_index) & (target_indices <= end_index)).astype(np.int32)
        return labels

    def get_test_target_global_indices(self) -> np.ndarray:
        """Return global target indices for each test window in the downsampled series."""
        start_index = int(self.n_train_raw + self.target_index_offset)
        if self.use_differencing:
            start_index += 1
        return np.arange(start_index, start_index + len(self.test_X), dtype=np.int32)

    def _load_uuv(self, file_path: str, downsample: int) -> Tuple[np.ndarray, np.ndarray]:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"UUV file not found: {p}")

        df = pd.read_csv(p, sep=r"\t+|\s+", engine="python")
        df.columns = [str(c).strip() for c in df.columns]

        if "Time" not in df.columns or "IMU_Yaw_Data" not in df.columns:
            raise ValueError(
                f"Expected columns ['Time', 'IMU_Yaw_Data']; found: {list(df.columns)}"
            )

        df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
        df["IMU_Yaw_Data"] = pd.to_numeric(df["IMU_Yaw_Data"], errors="coerce")
        df = df.dropna(subset=["Time", "IMU_Yaw_Data"]).reset_index(drop=True)

        if downsample > 1:
            # Interpolation-based resampling (legacy-style):
            # instead of taking every N-th point, preserve the full time span
            # and resample to approximately len(df)/downsample points.
            num_samples = max(2, int(np.ceil(len(df) / float(downsample))))
            original_time = df["Time"].values.astype(np.float32)
            target_time = np.linspace(original_time[0], original_time[-1], num_samples, dtype=np.float32)
            interpolated_yaw = np.interp(target_time, original_time, df["IMU_Yaw_Data"].values.astype(np.float32))

            # Keep a simple normalized index-like timeline for downstream plotting,
            # matching prior script behavior.
            t = np.linspace(0, num_samples - 1, num_samples, dtype=np.float32)
            x = interpolated_yaw.reshape(-1, 1).astype(np.float32)
            return t, x

        t = df["Time"].values.astype(np.float32)
        x = df[["IMU_Yaw_Data"]].values.astype(np.float32)
        return t, x

    def _build_windows(
        self,
        X: np.ndarray,
        window_size: int,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_idx = window_size + horizon - 1
        n = len(X)
        if n <= target_idx:
            raise ValueError("Not enough samples for windows")

        n_w = n - target_idx
        Xw = np.asarray([X[i : i + window_size] for i in range(n_w)], dtype=np.float32)
        y = np.asarray([X[i + target_idx] for i in range(n_w)], dtype=np.float32)
        return Xw, y

    def get_train_loader(self) -> DataLoader:
        ds = TensorDataset(self.train_X, self.train_y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self) -> DataLoader:
        ds = TensorDataset(self.test_X, self.test_y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    def get_train_data(self):
        return self.train_X, self.train_y

    def get_stats(self) -> Dict[str, float]:
        return {
            "dataset": "uuv_yaw",
            "file_path": self.file_path,
            "downsample": self.downsample,
            "window_size": self.window_size,
            "horizon": self.horizon,
            "use_differencing": bool(self.use_differencing),
            "train_ratio": self.train_ratio,
            "n_total_raw": int(self.n_total_raw),
            "n_train_raw": int(self.n_train_raw),
            "n_train_windows": int(len(self.train_X)),
            "n_test_windows": int(len(self.test_X)),
            "n_features": int(self.train_X.shape[-1]),
        }
