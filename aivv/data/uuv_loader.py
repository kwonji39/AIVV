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

        t, yaw = self._read_uuv_series(p)

        if len(t) != len(yaw):
            raise ValueError(
                f"Length mismatch between time ({len(t)}) and yaw ({len(yaw)}) in {p}"
            )
        if len(yaw) < 2:
            raise ValueError(f"Need at least 2 samples, got {len(yaw)} from {p}")

        # Ensure monotonic time for interpolation.
        order = np.argsort(t)
        t = t[order]
        yaw = yaw[order]
        t_unique, idx_unique = np.unique(t, return_index=True)
        yaw_unique = yaw[idx_unique]

        if downsample > 1:
            # Interpolation-based resampling (legacy-style):
            # Preserve the full time span and resample to approximately len(yaw)/downsample points.
            num_samples = max(2, int(np.ceil(len(yaw_unique) / float(downsample))))
            target_time = np.linspace(t_unique[0], t_unique[-1], num_samples, dtype=np.float32)
            interpolated_yaw = np.interp(target_time, t_unique, yaw_unique)

            # Keep a simple normalized index-like timeline for downstream plotting,
            # matching prior script behavior.
            t = np.linspace(0, num_samples - 1, num_samples, dtype=np.float32)
            x = interpolated_yaw.reshape(-1, 1).astype(np.float32)
            return t, x

        t = t_unique.astype(np.float32)
        x = yaw_unique.reshape(-1, 1).astype(np.float32)
        return t, x

    def _read_uuv_series(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        suffix = file_path.suffix.lower()
        if suffix in {".npy", ".npz"}:
            return self._read_uuv_numpy(file_path)
        return self._read_uuv_text(file_path)

    def _read_uuv_text(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(file_path, sep=r"\t+|\s+", engine="python")
        df.columns = [str(c).strip() for c in df.columns]

        if "Time" in df.columns and "IMU_Yaw_Data" in df.columns:
            time_vals = pd.to_numeric(df["Time"], errors="coerce")
            yaw_vals = pd.to_numeric(df["IMU_Yaw_Data"], errors="coerce")
        else:
            numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if numeric_df.shape[1] >= 2:
                time_vals = numeric_df.iloc[:, 0]
                yaw_vals = numeric_df.iloc[:, 1]
            elif numeric_df.shape[1] == 1:
                yaw_vals = numeric_df.iloc[:, 0]
                time_vals = pd.Series(np.arange(len(yaw_vals), dtype=np.float32))
            else:
                raise ValueError(
                    f"Could not parse numeric columns from text file: {file_path}"
                )

        parsed = pd.DataFrame({"time": time_vals, "yaw": yaw_vals}).dropna().reset_index(drop=True)
        if parsed.empty:
            raise ValueError(f"No valid (time, yaw) samples found in {file_path}")
        return (
            parsed["time"].to_numpy(dtype=np.float32),
            parsed["yaw"].to_numpy(dtype=np.float32),
        )

    def _read_uuv_numpy(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        def _is_monotonic(vec: np.ndarray) -> bool:
            if len(vec) < 2:
                return True
            diff = np.diff(vec)
            return bool(np.all(diff >= 0) or np.all(diff <= 0))

        def _signal_from_matrix(mat: np.ndarray) -> np.ndarray:
            # Prefer first non-constant feature; fallback to column 0.
            if mat.shape[1] == 1:
                return mat[:, 0].astype(np.float32)
            stds = np.nanstd(mat, axis=0)
            candidates = np.where(stds > 1e-12)[0]
            col = int(candidates[0]) if len(candidates) else 0
            return mat[:, col].astype(np.float32)

        payload = np.load(file_path, allow_pickle=True)

        if isinstance(payload, np.lib.npyio.NpzFile):
            keys = list(payload.keys())
            key_map = {k.lower(): k for k in keys}
            time_key = key_map.get("time") or key_map.get("t") or key_map.get("timestamps")
            yaw_key = (
                key_map.get("imu_yaw_data")
                or key_map.get("yaw")
                or key_map.get("y")
                or key_map.get("values")
                or key_map.get("signal")
                or key_map.get("data")
            )

            if yaw_key is None and len(keys) == 1:
                yaw_key = keys[0]

            if yaw_key is None:
                raise ValueError(
                    f".npz must include one of [imu_yaw_data, yaw, y, values, signal, data]. Keys: {keys}"
                )

            yaw_arr = np.asarray(payload[yaw_key], dtype=np.float32)
            if yaw_arr.ndim == 2:
                yaw = _signal_from_matrix(yaw_arr)
            else:
                yaw = yaw_arr.reshape(-1)
            if time_key is not None:
                time_vals = np.asarray(payload[time_key], dtype=np.float32).reshape(-1)
            else:
                time_vals = np.arange(len(yaw), dtype=np.float32)
            return time_vals, yaw

        arr = np.asarray(payload)
        if arr.dtype == object and arr.size == 1:
            maybe_dict = arr.item()
            if isinstance(maybe_dict, dict):
                key_map = {str(k).lower(): k for k in maybe_dict.keys()}
                yaw_key = (
                    key_map.get("imu_yaw_data")
                    or key_map.get("yaw")
                    or key_map.get("y")
                    or key_map.get("values")
                    or key_map.get("signal")
                    or key_map.get("data")
                )
                time_key = key_map.get("time") or key_map.get("t") or key_map.get("timestamps")
                if yaw_key is None:
                    raise ValueError(
                        f"Dict-like .npy must contain yaw key (e.g., 'yaw' or 'IMU_Yaw_Data'). Keys: {list(maybe_dict.keys())}"
                    )
                yaw_arr = np.asarray(maybe_dict[yaw_key], dtype=np.float32)
                if yaw_arr.ndim == 2:
                    yaw = _signal_from_matrix(yaw_arr)
                else:
                    yaw = yaw_arr.reshape(-1)
                if time_key is not None:
                    time_vals = np.asarray(maybe_dict[time_key], dtype=np.float32).reshape(-1)
                else:
                    time_vals = np.arange(len(yaw), dtype=np.float32)
                return time_vals, yaw

        if arr.ndim == 1:
            yaw = arr.astype(np.float32)
            time_vals = np.arange(len(yaw), dtype=np.float32)
            return time_vals, yaw

        if arr.ndim == 2:
            # Common case A: [N, 2+] with explicit time in col0 and signal in col1.
            # For multivariate feature matrices (e.g., satellite channels, [N, 25]/[N, 55]),
            # col0 is not time; use index-based time and select a non-constant feature.
            if arr.shape[1] >= 2:
                maybe_time = arr[:, 0].astype(np.float32)
                maybe_signal = arr[:, 1].astype(np.float32)
                if _is_monotonic(maybe_time) and float(np.nanstd(maybe_signal)) > 1e-12:
                    return maybe_time, maybe_signal
                yaw = _signal_from_matrix(arr)
                time_vals = np.arange(len(yaw), dtype=np.float32)
                return time_vals, yaw
            # Common case B: [N, 1] yaw only.
            if arr.shape[1] == 1:
                yaw = arr[:, 0].astype(np.float32)
                time_vals = np.arange(len(yaw), dtype=np.float32)
                return time_vals, yaw
            # Common case C: [2, N] where row0=time and row1=yaw.
            if arr.shape[0] == 2 and arr.shape[1] > 2:
                row0 = arr[0, :].astype(np.float32)
                row1 = arr[1, :].astype(np.float32)
                if _is_monotonic(row0) and float(np.nanstd(row1)) > 1e-12:
                    return row0, row1
                yaw = row0
                time_vals = np.arange(len(yaw), dtype=np.float32)
                return time_vals, yaw

        raise ValueError(
            f"Unsupported numpy payload shape for UUV data: {arr.shape}. "
            "Expected 1D yaw, [N,1], [N,2+], [2,N], dict-like, or .npz with yaw/time keys."
        )

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
