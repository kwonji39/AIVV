"""
UUV Engine

Main engine class that wraps the LSTM model and conformal predictor.
This is the shared resource that all agents interact with.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
import copy
import torch.nn as nn

from ..config import ACAConfig


class VariationalLSTM(nn.Module):
    """Variational LSTM with MC Dropout uncertainty."""

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout

        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = torch.device("xpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mc_dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()

        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b = x.size(0)
        x = self.relu(self.input_proj(x))

        if hidden is None:
            h0 = torch.zeros(self.num_layers, b, self.hidden_dim, device=self.device)
            c0 = torch.zeros(self.num_layers, b, self.hidden_dim, device=self.device)
            hidden = (h0, c0)

        lstm_out, hidden = self.lstm(x, hidden)
        last = lstm_out[:, -1, :]

        out = self.mc_dropout(last)
        out = self.relu(self.fc1(out))
        out = self.mc_dropout(out)
        out = self.fc2(out)
        return out, hidden

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.train()
        preds = []
        try:
            with torch.no_grad():
                for _ in range(n_samples):
                    p, _ = self.forward(x)
                    preds.append(p)
            preds = torch.stack(preds, dim=0)
            return preds.mean(dim=0), preds.std(dim=0)
        finally:
            self.train(was_training)

    def clone(self) -> "VariationalLSTM":
        cloned = VariationalLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout_rate,
            device=self.device,
        )
        cloned.load_state_dict(copy.deepcopy(self.state_dict()))
        return cloned

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMTrainer:
    def __init__(self, model: VariationalLSTM, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        pred, _ = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def fine_tune(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 5,
        lr: Optional[float] = None,
    ) -> list:
        if lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr

        losses = []
        for _ in range(epochs):
            losses.append(self.train_step(x, y))
        return losses


class ConformalPredictor:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None
        self.is_calibrated = False

    def calibrate(self, predictions: Union[np.ndarray, torch.Tensor], actuals: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(actuals, torch.Tensor):
            actuals = actuals.detach().cpu().numpy()

        p = np.asarray(predictions).flatten()
        a = np.asarray(actuals).flatten()
        self.calibration_scores = np.abs(p - a)
        self.is_calibrated = True

    def get_interval_width(self, alpha: Optional[float] = None) -> float:
        if not self.is_calibrated:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")
        alpha = self.alpha if alpha is None else alpha
        n = len(self.calibration_scores)
        q = np.ceil((n + 1) * (1 - alpha)) / n
        q = min(q, 1.0)
        return float(np.quantile(self.calibration_scores, q))

    def check_conformity(self, prediction: float, actual: float, alpha: Optional[float] = None) -> Tuple[bool, float]:
        alpha = self.alpha if alpha is None else alpha
        w = self.get_interval_width(alpha)
        err = abs(float(prediction) - float(actual))
        return err <= w, float(err)

    def recalibrate(
        self,
        new_predictions: Union[np.ndarray, torch.Tensor],
        new_actuals: Union[np.ndarray, torch.Tensor],
        weight_new: float = 0.3,
    ) -> None:
        if isinstance(new_predictions, torch.Tensor):
            new_predictions = new_predictions.detach().cpu().numpy()
        if isinstance(new_actuals, torch.Tensor):
            new_actuals = new_actuals.detach().cpu().numpy()

        new_scores = np.abs(np.asarray(new_predictions).flatten() - np.asarray(new_actuals).flatten())

        if self.is_calibrated:
            n_old = len(self.calibration_scores)
            n_new = len(new_scores)
            n_keep_old = int(n_old * (1 - weight_new))
            n_keep_new = int(n_new * weight_new) if weight_new > 0 else n_new

            if n_keep_old > 0:
                old_sample = np.random.choice(self.calibration_scores, size=min(n_keep_old, n_old), replace=False)
                self.calibration_scores = np.concatenate([old_sample, new_scores[-n_keep_new:]])
            else:
                self.calibration_scores = new_scores
        else:
            self.calibration_scores = new_scores
            self.is_calibrated = True

    def clone(self) -> "ConformalPredictor":
        cloned = ConformalPredictor(alpha=self.alpha)
        if self.is_calibrated:
            cloned.calibration_scores = self.calibration_scores.copy()
            cloned.is_calibrated = True
        return cloned

    def update_alpha(self, new_alpha: float) -> None:
        assert 0 < new_alpha < 1, "Alpha must be between 0 and 1"
        self.alpha = float(new_alpha)

    def get_stats(self) -> dict:
        if not self.is_calibrated:
            return {"calibrated": False}
        return {
            "calibrated": True,
            "n_samples": len(self.calibration_scores),
            "mean_score": float(np.mean(self.calibration_scores)),
            "std_score": float(np.std(self.calibration_scores)),
            "median_score": float(np.median(self.calibration_scores)),
            "max_score": float(np.max(self.calibration_scores)),
            "current_alpha": self.alpha,
            "current_interval": self.get_interval_width(),
        }


class UncertaintyCalibrator:
    """Calibrates an uncertainty (sigma) threshold from calibration data."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.uncertainty_scores: Optional[np.ndarray] = None
        self.is_calibrated = False

    def calibrate(self, sigmas: Union[np.ndarray, torch.Tensor]) -> None:
        if isinstance(sigmas, torch.Tensor):
            sigmas = sigmas.detach().cpu().numpy()
        s = np.asarray(sigmas).flatten()
        self.uncertainty_scores = np.maximum(s, 0.0)
        self.is_calibrated = True

    def get_threshold(self, alpha: Optional[float] = None) -> float:
        if not self.is_calibrated:
            raise ValueError("Uncertainty calibrator not calibrated. Call calibrate() first.")
        alpha = self.alpha if alpha is None else alpha
        n = len(self.uncertainty_scores)
        q = np.ceil((n + 1) * (1 - alpha)) / n
        q = min(q, 1.0)
        return float(np.quantile(self.uncertainty_scores, q))

    def clone(self) -> "UncertaintyCalibrator":
        cloned = UncertaintyCalibrator(alpha=self.alpha)
        if self.is_calibrated:
            cloned.uncertainty_scores = self.uncertainty_scores.copy()
            cloned.is_calibrated = True
        return cloned

    def update_alpha(self, new_alpha: float) -> None:
        assert 0 < new_alpha < 1, "Alpha must be between 0 and 1"
        self.alpha = float(new_alpha)

    def get_stats(self) -> dict:
        if not self.is_calibrated:
            return {"calibrated": False}
        return {
            "calibrated": True,
            "n_samples": len(self.uncertainty_scores),
            "mean_sigma": float(np.mean(self.uncertainty_scores)),
            "std_sigma": float(np.std(self.uncertainty_scores)),
            "median_sigma": float(np.median(self.uncertainty_scores)),
            "max_sigma": float(np.max(self.uncertainty_scores)),
            "current_alpha": self.alpha,
            "current_threshold": self.get_threshold(),
        }


class UUVEngine:
    """
    The Mathematical Engine for UUV anomaly detection.
    
    NOT an agent - this is a Python class that provides the computational
    core. All agents interact with this single instance (or clones of it).
    
    Methods (Tools for Agents):
        - predict(window): Returns prediction_mean, uncertainty_sigma
        - get_conformal_interval(alpha): Returns prediction interval width
        - fine_tune(history_window, lr, epochs): Updates LSTM weights
    """
    
    def __init__(
        self,
        config: Optional[ACAConfig] = None,
        model: Optional[VariationalLSTM] = None,
        conformal: Optional[ConformalPredictor] = None,
        uncertainty_calibrator: Optional[UncertaintyCalibrator] = None,
    ):
        """
        Initialize the UUV Engine.
        
        Args:
            config: ACA configuration
            model: Optional pre-trained LSTM model
            conformal: Optional calibrated conformal predictor
        """
        self.config = config or ACAConfig()
        
        # Initialize or use provided model
        if model is not None:
            self.model = model
        else:
            self.model = VariationalLSTM(
                input_dim=self.config.lstm_input_dim,
                hidden_dim=self.config.lstm_hidden_dim,
                num_layers=self.config.lstm_num_layers,
                dropout=self.config.lstm_dropout,
                device=self.config.get_device()
            )
        
        # Initialize trainer
        self.trainer = LSTMTrainer(
            self.model,
            learning_rate=self.config.fine_tune_lr
        )
        
        # Initialize or use provided conformal predictor
        if conformal is not None:
            self.conformal = conformal
        else:
            self.conformal = ConformalPredictor(alpha=self.config.default_alpha)

        # Initialize or use provided uncertainty calibrator
        if uncertainty_calibrator is not None:
            self.uncertainty_calibrator = uncertainty_calibrator
        else:
            self.uncertainty_calibrator = UncertaintyCalibrator(alpha=self.config.default_alpha)
        
        # State tracking
        self.is_trained = False
        self.prediction_history: list = []
        self.device = self.config.get_device()
        self.training_evidence: Dict[str, Any] = {}
        self.training_evidence_revision: int = 0
        self._training_reference: Dict[str, Any] = {}

    def _to_cpu_tensor(self, value: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        return torch.as_tensor(value, dtype=torch.float32).cpu()

    def _predict_no_tracking(self, window: torch.Tensor) -> Tuple[float, float]:
        if window.dim() == 2:
            window = window.unsqueeze(0)
        window = window.to(self.device)
        mean, std = self.model.predict_with_uncertainty(
            window,
            n_samples=self.config.mc_samples,
        )
        return float(mean.squeeze().detach().cpu().numpy()), float(std.squeeze().detach().cpu().numpy())

    def _quantile_summary(self, values: np.ndarray) -> Dict[str, float]:
        if values.size == 0:
            return {}
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'q50': float(np.quantile(values, 0.50)),
            'q90': float(np.quantile(values, 0.90)),
            'q95': float(np.quantile(values, 0.95)),
            'q99': float(np.quantile(values, 0.99)),
        }

    def _build_raw_window_baseline(self, raw_train_series: Optional[np.ndarray]) -> Dict[str, Any]:
        if raw_train_series is None:
            return {}

        raw = np.asarray(raw_train_series, dtype=np.float32).reshape(-1)
        if raw.size == 0:
            return {}

        abs_values = np.abs(raw)
        step_abs = np.abs(np.diff(raw)) if raw.size >= 2 else np.asarray([], dtype=np.float32)

        window_size = int(getattr(self.config, 'window_size', 1))
        windows = []
        if raw.size >= window_size and window_size > 0:
            for i in range(raw.size - window_size + 1):
                windows.append(raw[i:i + window_size])
        window_arr = np.asarray(windows, dtype=np.float32) if windows else np.empty((0, window_size), dtype=np.float32)

        if len(window_arr) > 0:
            window_range = np.ptp(window_arr, axis=1)
            window_std = np.std(window_arr, axis=1)
            window_delta_last_first_abs = np.abs(window_arr[:, -1] - window_arr[:, 0])
            window_max_abs_step = np.max(np.abs(np.diff(window_arr, axis=1)), axis=1) if window_size >= 2 else np.zeros((len(window_arr),), dtype=np.float32)

            med_idx = int(np.argmin(np.abs(window_range - np.median(window_range))))
            hi_step_idx = int(np.argmax(window_max_abs_step))
            hi_range_idx = int(np.argmax(window_range))

            representative_windows = {
                'median_range_window': [float(v) for v in window_arr[med_idx].tolist()],
                'high_step_window': [float(v) for v in window_arr[hi_step_idx].tolist()],
                'high_range_window': [float(v) for v in window_arr[hi_range_idx].tolist()],
            }
        else:
            window_range = np.asarray([], dtype=np.float32)
            window_std = np.asarray([], dtype=np.float32)
            window_delta_last_first_abs = np.asarray([], dtype=np.float32)
            window_max_abs_step = np.asarray([], dtype=np.float32)
            representative_windows = {}

        return {
            'series_length': int(raw.size),
            'value_distribution': self._quantile_summary(raw),
            'abs_value_distribution': self._quantile_summary(abs_values),
            'step_abs_distribution': self._quantile_summary(step_abs),
            'window_range_distribution': self._quantile_summary(window_range),
            'window_std_distribution': self._quantile_summary(window_std),
            'window_delta_last_first_abs_distribution': self._quantile_summary(window_delta_last_first_abs),
            'window_max_abs_step_distribution': self._quantile_summary(window_max_abs_step),
            'representative_windows': representative_windows,
        }

    def _build_probe_summary(
        self,
        windows: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
        label: str,
        max_samples: int,
    ) -> Dict[str, Any]:
        if windows is None or targets is None:
            return {}
        if len(windows) == 0:
            return {}

        n_total = int(len(windows))
        n_probe = min(max_samples, n_total)
        idx = np.linspace(0, n_total - 1, num=n_probe, dtype=int)

        errors = []
        sigmas = []
        preds = []
        actuals = []
        for i in idx.tolist():
            pred, sigma = self._predict_no_tracking(windows[i:i + 1])
            actual = float(targets[i].reshape(-1)[0].item())
            preds.append(pred)
            actuals.append(actual)
            errors.append(abs(pred - actual))
            sigmas.append(sigma)

        errors_arr = np.asarray(errors, dtype=np.float32)
        sigmas_arr = np.asarray(sigmas, dtype=np.float32)
        interval_95 = self.get_conformal_interval(0.05)
        uncertainty_95 = self.get_uncertainty_threshold(0.05)

        return {
            'label': label,
            'n_total_samples': n_total,
            'n_probe_samples': int(n_probe),
            'error_distribution': self._quantile_summary(errors_arr),
            'uncertainty_distribution': self._quantile_summary(sigmas_arr),
            'pass_rate_at_95': float(np.mean(errors_arr <= interval_95)) if np.isfinite(interval_95) else None,
            'uncertainty_within_95_rate': float(np.mean(sigmas_arr <= uncertainty_95)) if np.isfinite(uncertainty_95) else None,
            'sample_pairs': [
                {
                    'prediction': float(preds[j]),
                    'actual': float(actuals[j]),
                    'error': float(errors_arr[j]),
                    'uncertainty': float(sigmas_arr[j]),
                }
                for j in range(min(5, len(preds)))
            ],
        }

    def register_training_reference(
        self,
        train_windows: Union[np.ndarray, torch.Tensor],
        train_targets: Union[np.ndarray, torch.Tensor],
        calibration_windows: Optional[Union[np.ndarray, torch.Tensor]] = None,
        calibration_targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        raw_train_series: Optional[Union[np.ndarray, torch.Tensor]] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
        scaler_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register train/calibration references used to build evidence for council agents."""
        self._training_reference = {
            'train_windows': self._to_cpu_tensor(train_windows),
            'train_targets': self._to_cpu_tensor(train_targets),
            'calibration_windows': None if calibration_windows is None else self._to_cpu_tensor(calibration_windows),
            'calibration_targets': None if calibration_targets is None else self._to_cpu_tensor(calibration_targets),
            'raw_train_series': None if raw_train_series is None else np.asarray(raw_train_series, dtype=np.float32).copy(),
            'dataset_stats': copy.deepcopy(dataset_stats or {}),
            'scaler_stats': copy.deepcopy(scaler_stats or {}),
        }
        return self.refresh_training_evidence(reason='initial_training')

    def refresh_training_evidence(self, reason: str = 'model_update') -> Dict[str, Any]:
        """Recompute training-derived evidence for council agents using the current model state."""
        if not self._training_reference:
            self.training_evidence = {}
            return {}

        probe_samples = 32
        ref = self._training_reference
        next_revision = int(self.training_evidence_revision) + 1
        evidence = {
            'revision': next_revision,
            'reason': reason,
            'model_state': {
                'is_trained': bool(self.is_trained),
                'device': str(self.device),
                'operating_alpha': float(self.conformal.alpha),
                'lstm_hidden_dim': int(self.model.hidden_dim),
                'lstm_num_layers': int(self.model.num_layers),
                'dropout': float(self.model.dropout_rate),
                'mc_samples': int(self.config.mc_samples),
                'parameter_count': int(self.model.get_num_parameters()),
            },
            'dataset_stats': copy.deepcopy(ref.get('dataset_stats', {})),
            'scaler_stats': copy.deepcopy(ref.get('scaler_stats', {})),
            'conformal_stats': self.conformal.get_stats(),
            'uncertainty_stats': self.uncertainty_calibrator.get_stats(),
            'raw_window_baseline': self._build_raw_window_baseline(ref.get('raw_train_series')),
            'model_baseline': {
                'train_probe': self._build_probe_summary(
                    ref.get('train_windows'),
                    ref.get('train_targets'),
                    label='train',
                    max_samples=probe_samples,
                ),
                'calibration_probe': self._build_probe_summary(
                    ref.get('calibration_windows'),
                    ref.get('calibration_targets'),
                    label='calibration',
                    max_samples=probe_samples,
                ),
            },
        }
        self.training_evidence = evidence
        self.training_evidence_revision = next_revision
        return copy.deepcopy(evidence)

    def get_training_evidence(self) -> Dict[str, Any]:
        """Return a deep copy of current training-derived evidence."""
        return copy.deepcopy(self.training_evidence)
    
    def predict(
        self,
        window: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[float, float]:
        """
        Make prediction with uncertainty estimation.
        
        Used by: Agent 1 (Sentry), Agent 3 (Tuner)
        
        Args:
            window: Input window of shape (seq_len, features) or (batch, seq_len, features)
            
        Returns:
            prediction_mean: Mean prediction value
            uncertainty_sigma: Standard deviation (uncertainty)
        """
        # Convert to tensor if needed
        if isinstance(window, np.ndarray):
            window = torch.FloatTensor(window)
        
        # Ensure correct shape (batch, seq_len, features)
        if window.dim() == 2:
            window = window.unsqueeze(0)
        
        # Move to device
        window = window.to(self.device)
        
        # Get prediction with uncertainty
        mean, std = self.model.predict_with_uncertainty(
            window,
            n_samples=self.config.mc_samples
        )
        
        # Extract scalar values
        prediction_mean = float(mean.squeeze().cpu().numpy())
        uncertainty_sigma = float(std.squeeze().cpu().numpy())
        
        # Track prediction
        self.prediction_history.append({
            'mean': prediction_mean,
            'sigma': uncertainty_sigma
        })
        
        return prediction_mean, uncertainty_sigma
    
    def get_conformal_interval(self, alpha: Optional[float] = None) -> float:
        """
        Get the prediction interval width for a given confidence level.
        
        Used by: Agent 1 (Sentry), Agent 3 (Tuner)
        
        Args:
            alpha: Significance level (e.g., 0.05 for 95% confidence)
                   Uses default if not specified.
                   
        Returns:
            interval_width: The width of the prediction interval
        """
        if alpha is None:
            alpha = self.config.default_alpha
        
        if not self.conformal.is_calibrated:
            # Return a default value if not calibrated
            return float('inf')
        
        return self.conformal.get_interval_width(alpha)

    def calibrate_uncertainty(
        self,
        sigmas: Union[np.ndarray, torch.Tensor],
        alpha: Optional[float] = None
    ) -> None:
        """Calibrate uncertainty threshold from sigma values."""
        if alpha is not None:
            self.uncertainty_calibrator.alpha = alpha
        self.uncertainty_calibrator.calibrate(sigmas)

    def get_uncertainty_threshold(self, alpha: Optional[float] = None) -> float:
        """Get calibrated uncertainty threshold for a given alpha."""
        if alpha is None:
            alpha = self.config.default_alpha
        if not self.uncertainty_calibrator.is_calibrated:
            return float('inf')
        return self.uncertainty_calibrator.get_threshold(alpha)
    
    def fine_tune(
        self,
        history_window: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        lr: Optional[float] = None,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Update LSTM weights via backpropagation.
        
        Used by: Agent 3 (Tuner)
        
        Args:
            history_window: Training sequences (batch, seq_len, features)
            targets: Target values (batch, 1)
            lr: Learning rate (uses config default if not specified)
            epochs: Number of epochs (uses config default if not specified)
            verbose: Print epoch progress
            
        Returns:
            Dict with training info (losses, final_loss)
        """
        lr = lr or self.config.fine_tune_lr
        epochs = epochs or self.config.fine_tune_epochs
        
        # Convert to tensors
        if isinstance(history_window, np.ndarray):
            history_window = torch.FloatTensor(history_window)
        if isinstance(targets, np.ndarray):
            targets = torch.FloatTensor(targets)
        
        # Ensure correct shape
        if history_window.dim() == 2:
            history_window = history_window.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        # Move to device
        history_window = history_window.to(self.device)
        targets = targets.to(self.device)
        
        # Fine-tune with epoch logging
        losses = []
        for epoch in range(epochs):
            epoch_loss = self.trainer.fine_tune(
                history_window,
                targets,
                epochs=1,
                lr=lr
            )
            losses.extend(epoch_loss)
            if verbose:
                print(f"      Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss[-1]:.6f}", flush=True)
            
        self.is_trained = True
        
        return {
            'losses': losses,
            'final_loss': losses[-1] if losses else None,
            'epochs': epochs,
            'lr': lr
        }
    
    def calibrate_conformal(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        actuals: Union[np.ndarray, torch.Tensor],
        alpha: Optional[float] = None
    ) -> None:
        """
        Calibrate the conformal predictor with calibration data.
        
        Args:
            predictions: Model predictions on calibration set
            actuals: True values on calibration set
            alpha: Optional confidence level (default uses config value)
        """
        if alpha is not None:
            self.conformal.alpha = alpha
        self.conformal.calibrate(predictions, actuals)

    def set_operating_alpha(self, alpha: float) -> float:
        """Set current conformal operating alpha and return applied value."""
        self.conformal.update_alpha(float(alpha))
        return float(self.conformal.alpha)
    
    def recalibrate_conformal(
        self,
        new_alpha: Optional[float] = None,
        new_predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        new_actuals: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Recalibrate conformal predictor (adjust alpha or add new data).
        
        Used by: Agent 3 (Tuner) for noise handling
        
        Args:
            new_alpha: New significance level (e.g., 0.01 for 99% confidence)
            new_predictions: Optional new calibration predictions
            new_actuals: Optional new calibration actuals
            
        Returns:
            Dict with old and new interval widths
        """
        old_alpha = self.conformal.alpha
        old_width = self.get_conformal_interval()
        
        if new_predictions is not None and new_actuals is not None:
            self.conformal.recalibrate(new_predictions, new_actuals)
        
        if new_alpha is not None:
            self.conformal.update_alpha(new_alpha)
        
        new_width = self.get_conformal_interval()
        
        return {
            'old_alpha': old_alpha,
            'new_alpha': self.conformal.alpha,
            'old_width': old_width,
            'new_width': new_width
        }
    
    def check_anomaly(
        self,
        prediction: float,
        actual: float,
        alpha: Optional[float] = None
    ) -> Tuple[bool, float, float]:
        """
        Check if an observation is anomalous.
        
        Args:
            prediction: Model prediction
            actual: Actual value
            alpha: Significance level
            
        Returns:
            is_anomaly: True if |error| > interval_width
            error: Absolute error
            interval_width: Current interval width
        """
        if alpha is None:
            alpha = self.config.default_alpha
        
        is_conforming, error = self.conformal.check_conformity(
            prediction, actual, alpha
        )
        interval_width = self.get_conformal_interval(alpha)
        
        return not is_conforming, error, interval_width
    
    def clone(self) -> 'UUVEngine':
        """
        Create a deep copy of the engine for simulation.
        
        Used by: Agent 3 (Tuner) for testing adjustments
        
        Returns:
            A new UUVEngine with copied model and conformal predictor
        """
        cloned = UUVEngine(
            config=self.config,  # Share config (immutable settings)
            model=self.model.clone(),
            conformal=self.conformal.clone(),
            uncertainty_calibrator=self.uncertainty_calibrator.clone(),
        )
        cloned.is_trained = self.is_trained
        cloned.prediction_history = self.prediction_history.copy()
        cloned.training_evidence = copy.deepcopy(self.training_evidence)
        cloned.training_evidence_revision = self.training_evidence_revision
        cloned._training_reference = copy.deepcopy(self._training_reference)
        return cloned
    
    def promote_from(self, temp_engine: 'UUVEngine') -> None:
        """
        Update this engine's state from a temporary engine.
        
        Used when Council approves Agent 3's changes.
        
        Args:
            temp_engine: The temporary engine with approved changes
        """
        self.model.load_state_dict(temp_engine.model.state_dict())
        self.conformal = temp_engine.conformal.clone()
        self.uncertainty_calibrator = temp_engine.uncertainty_calibrator.clone()
        self.is_trained = temp_engine.is_trained
        self.training_evidence = copy.deepcopy(temp_engine.training_evidence)
        self.training_evidence_revision = int(temp_engine.training_evidence_revision)
        self._training_reference = copy.deepcopy(temp_engine._training_reference)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current engine state for logging/debugging."""
        return {
            'is_trained': self.is_trained,
            'model_params': self.model.get_num_parameters(),
            'conformal_calibrated': self.conformal.is_calibrated,
            'conformal_stats': self.conformal.get_stats(),
            'uncertainty_calibrated': self.uncertainty_calibrator.is_calibrated,
            'uncertainty_stats': self.uncertainty_calibrator.get_stats(),
            'training_evidence_revision': self.training_evidence_revision,
            'device': str(self.device),
            'prediction_count': len(self.prediction_history)
        }
    
    def save(self, path: str) -> None:
        """Save engine state to disk."""
        state = {
            'model_state': self.model.state_dict(),
            'conformal_scores': self.conformal.calibration_scores,
            'conformal_alpha': self.conformal.alpha,
            'is_trained': self.is_trained,
            'training_evidence': self.training_evidence,
            'training_evidence_revision': self.training_evidence_revision,
        }
        torch.save(state, path)
    
    def load(self, path: str) -> None:
        """Load engine state from disk."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.conformal.calibration_scores = state['conformal_scores']
        self.conformal.alpha = state['conformal_alpha']
        self.conformal.is_calibrated = True
        self.is_trained = state['is_trained']
        self.training_evidence = copy.deepcopy(state.get('training_evidence', {}))
        self.training_evidence_revision = int(state.get('training_evidence_revision', 0))



__all__ = [
    "VariationalLSTM",
    "LSTMTrainer",
    "ConformalPredictor",
    "UUVEngine",
]
