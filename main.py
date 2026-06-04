#!/usr/bin/env python
"""
AIVV - Main Entry Point

Run the full AIVV pipeline on UUV yaw data.

Usage:
    python main.py [--epochs N] [--samples N] [--data-path PATH]
    
Examples:
    python main.py --epochs 20
    python main.py --epochs 10 --samples 200
"""

import argparse
import atexit
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress verbose httpx/httpcore logging (only show errors)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from aivv.config import ACAConfig
from aivv.engine.engine import UUVEngine
from aivv.orchestrator import ACAOrchestrator
from aivv.data.uuv_loader import UUVDataLoader
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed_value: Optional[int]) -> None:
    """Set seeds for Python random, NumPy, and PyTorch for reproducibility."""
    if seed_value is None:
        return
    
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


class _TeeStream:
    """Mirror writes to console and a log file stream."""

    def __init__(self, console_stream, file_stream):
        self.console_stream = console_stream
        self.file_stream = file_stream

    def write(self, data):
        try:
            self.console_stream.write(data)
        except Exception:
            pass
        try:
            if not self.file_stream.closed:
                self.file_stream.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self.console_stream.flush()
        except Exception:
            pass
        try:
            if not self.file_stream.closed:
                self.file_stream.flush()
        except Exception:
            pass


def setup_run_terminal_logging(
    base_dir: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> Path:
    """Save terminal output to logs/run_YYYYMMDD_HHMMSS/terminal_output.log."""
    root = base_dir or Path(__file__).resolve().parent
    if run_dir is None:
        run_dir = root / "logs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "terminal_output.log"
    log_file = open(log_path, "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)

    def _cleanup_streams() -> None:
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        finally:
            if not log_file.closed:
                log_file.close()

    atexit.register(_cleanup_streams)
    return run_dir


def train_engine(
    engine: UUVEngine,
    train_loader,
    epochs: int = 10,
    cal_data: tuple = None
) -> None:
    """
    Train the engine and calibrate conformal predictor.
    
    Args:
        engine: UUVEngine instance
        train_loader: Training data loader
        epochs: Number of training epochs
        cal_data: Optional (cal_X, cal_y) tuple for conformal calibration
    """
    print("Training LSTM model...")
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            loss_info = engine.fine_tune(batch_x, batch_y, epochs=1, verbose=False)
            total_loss += loss_info['final_loss']
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")
    
    # Calibrate conformal predictor using CALIBRATION data (proper held-out set)
    if cal_data is not None:
        cal_X, cal_y = cal_data
        print(f"Calibrating conformal predictor on {len(cal_X)} calibration samples...")
        
        predictions = []
        sigmas = []
        with torch.no_grad():
            for i in range(len(cal_X)):
                pred, sigma = engine.predict(cal_X[i:i+1])
                predictions.append(pred)
                sigmas.append(sigma)
        
        predictions = np.array(predictions)
        sigmas = np.array(sigmas)
        actuals = cal_y.cpu().numpy().flatten()
        
        engine.calibrate_conformal(predictions, actuals)
        engine.calibrate_uncertainty(sigmas)
        print("Calibration complete (using held-out calibration set)!")
    else:
        # Fallback: calibrate on a subset from the training loader
        print("Calibration data not provided. Building fallback calibration set from training data...")
        train_X, train_y = train_loader.dataset.tensors
        n_cal = max(1, int(0.2 * len(train_X)))
        cal_X, cal_y = train_X[-n_cal:], train_y[-n_cal:]

        predictions = []
        sigmas = []
        with torch.no_grad():
            for i in range(len(cal_X)):
                pred, sigma = engine.predict(cal_X[i:i+1])
                predictions.append(pred)
                sigmas.append(sigma)

        predictions = np.array(predictions)
        sigmas = np.array(sigmas)
        actuals = cal_y.cpu().numpy().flatten()
        engine.calibrate_conformal(predictions, actuals)
        engine.calibrate_uncertainty(sigmas)
        print("Calibration complete (fallback training subset)!")


def apply_fault_persistence(flags: np.ndarray, min_consecutive: int) -> np.ndarray:
    """Require at least `min_consecutive` sequential flags to keep a fault on."""
    if min_consecutive <= 1:
        return flags.astype(bool)

    out = np.zeros_like(flags, dtype=bool)
    n = len(flags)
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if (j - i) >= min_consecutive:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def resolve_uuv_fault_window(file_path: str, start_index: Optional[int], end_index: Optional[int]) -> Optional[Tuple[int, int]]:
    """Resolve the explicit simulated-fault window for UUV evaluation."""
    if start_index is not None and end_index is not None:
        if end_index < start_index:
            raise ValueError("uuv fault end index must be >= start index")
        return int(start_index), int(end_index)

    if start_index is not None or end_index is not None:
        raise ValueError("uuv fault start/end indices must be provided together")

    name = Path(file_path).stem.lower()
    if "fault" in name and "normal" not in name:
        return 1200, 1250

    return None


def cumulative_prf(pred_labels: np.ndarray, true_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cumulative precision/recall/F1 up to each sample index."""
    n = len(pred_labels)
    p = np.zeros((n,), dtype=np.float32)
    r = np.zeros((n,), dtype=np.float32)
    f1 = np.zeros((n,), dtype=np.float32)

    tp = fp = fn = 0
    for i in range(n):
        pred = int(pred_labels[i])
        true = int(true_labels[i])
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1_i = 2 * prec * rec / max(1e-12, prec + rec)
        p[i], r[i], f1[i] = prec, rec, f1_i

    return p, r, f1


def save_inference_plots(trace: dict, out_dir: Path, fault_persistence: int = 1) -> None:
    """Save plots to track ACA behavior during inference."""
    t = np.asarray(trace["index"], dtype=np.int32)
    pred = np.asarray(trace["prediction"], dtype=np.float32)
    actual = np.asarray(trace["actual"], dtype=np.float32)
    err = np.asarray(trace["error"], dtype=np.float32)
    bound = np.asarray(trace["bound"], dtype=np.float32)
    uncertainty = np.asarray(trace.get("uncertainty", []), dtype=np.float32)
    uncertainty_bound = np.asarray(trace.get("uncertainty_bound", []), dtype=np.float32)
    init_fail = np.asarray(trace["initial_fail"], dtype=bool)
    raw_council_fail = np.asarray(trace.get("raw_council_fail", []), dtype=bool)
    final_fail = np.asarray(trace["final_fail"], dtype=bool)
    pred_label = np.asarray(trace.get("pred_label", []), dtype=np.int32)
    true_label = np.asarray(trace.get("true_label", []), dtype=np.float32)

    persistent_fail = apply_fault_persistence(final_fail, fault_persistence)

    # 1) Prediction vs actual
    finite_pred_band = np.isfinite(bound)
    lower = pred - bound
    upper = pred + bound
    plt.figure(figsize=(12, 4.2))
    plt.plot(t, actual, linewidth=1.0, label="actual")
    plt.plot(t, pred, linewidth=1.0, alpha=0.9, label="prediction")
    if finite_pred_band.any():
        plt.fill_between(
            t[finite_pred_band],
            lower[finite_pred_band],
            upper[finite_pred_band],
            alpha=0.22,
            label="calibrated prediction interval (target coverage=95.0%)",
        )
    plt.title("ACA inference: prediction vs actual")
    plt.xlabel("sample index")
    plt.ylabel("value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "aca_pred_vs_actual.png", dpi=160)
    plt.close()

    # 2) Error vs conformal bound
    finite = np.isfinite(bound)
    plt.figure(figsize=(12, 4.2))
    plt.plot(t, err, linewidth=1.0, label="|prediction - actual|")
    if finite.any():
        plt.plot(t[finite], bound[finite], linewidth=1.0, linestyle="--", label="conformal bound")
    if final_fail.any():
        plt.scatter(t[final_fail], err[final_fail], s=10, color="red", alpha=0.7, label="final FAIL")
    plt.title("ACA inference: residual vs conformal bound")
    plt.xlabel("sample index")
    plt.ylabel("error")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "aca_residual_vs_bound.png", dpi=160)
    plt.close()

    # 2b) Uncertainty vs calibrated uncertainty bound
    if len(uncertainty) == len(t) and len(uncertainty) > 0:
        finite_u = np.isfinite(uncertainty_bound)
        plt.figure(figsize=(12, 4.2))
        plt.plot(t, uncertainty, linewidth=1.0, label="uncertainty sigma")
        if finite_u.any():
            plt.plot(
                t[finite_u],
                uncertainty_bound[finite_u],
                linewidth=1.0,
                linestyle="--",
                label="uncertainty bound",
            )
        if final_fail.any():
            plt.scatter(t[final_fail], uncertainty[final_fail], s=10, color="red", alpha=0.7, label="final FAIL")
        plt.title("ACA inference: uncertainty vs uncertainty bound")
        plt.xlabel("sample index")
        plt.ylabel("sigma")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "aca_uncertainty_vs_bound.png", dpi=160)
        plt.close()

    # 3) Decision timeline (initial/final/persistent)
    plt.figure(figsize=(12, 3.8))
    plt.step(t, init_fail.astype(np.int32), where="post", linewidth=1.1, label="initial FAIL (Sentry)")
    if len(raw_council_fail) == len(t):
        plt.step(
            t,
            raw_council_fail.astype(np.int32),
            where="post",
            linewidth=1.0,
            alpha=0.85,
            linestyle=":",
            label="raw FAIL (Council before temporal filter)",
        )
    plt.step(t, final_fail.astype(np.int32), where="post", linewidth=1.1, label="final FAIL (Council)")
    plt.step(
        t,
        persistent_fail.astype(np.int32),
        where="post",
        linewidth=1.2,
        linestyle="--",
        label=f"persistent FAIL (k={fault_persistence})",
    )
    plt.yticks([0, 1], ["PASS", "FAIL"])
    plt.ylim(-0.2, 1.2)
    plt.title("ACA inference: decision timeline")
    plt.xlabel("sample index")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "aca_decision_timeline.png", dpi=160)
    plt.close()

    # 4) Cumulative precision / recall / F1 over time (if labels are present)
    if len(pred_label) == len(true_label) and len(pred_label) > 0 and np.all(np.isfinite(true_label)):
        true_label_int = true_label.astype(np.int32)
        p, r, f1 = cumulative_prf(pred_label, true_label_int)
        persistent_pred = apply_fault_persistence(final_fail, fault_persistence).astype(np.int32)
        p_p, r_p, f1_p = cumulative_prf(persistent_pred, true_label_int)

        plt.figure(figsize=(12, 4.2))
        plt.plot(t, p, linewidth=1.0, label="precision (final)")
        plt.plot(t, r, linewidth=1.0, label="recall (final)")
        plt.plot(t, f1, linewidth=1.2, label="F1 (final)")
        plt.plot(t, f1_p, linewidth=1.2, linestyle="--", label=f"F1 (persistent k={fault_persistence})")
        plt.ylim(0.0, 1.0)
        plt.xlabel("sample index")
        plt.ylabel("metric")
        plt.title("ACA inference: cumulative precision/recall/F1")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "aca_cumulative_prf1.png", dpi=160)
        plt.close()


def save_uuv_fdi_style_plot(
    trace: dict,
    data_loader: UUVDataLoader,
    out_dir: Path,
    alpha: float = 0.05,
    fault_persistence: int = 1,
    fail_key: str = "final_fail",
    output_filename: str = "aca_uuv_fdi_style.png",
    prediction_label: str = "Predicted Yaw",
    title: str = "FDI of Yaw Rate",
    fail_marker_mode: str = "trigger",
) -> None:
    """Save a UUV FDI-style plot similar to the standalone script output.

    Plot elements:
    - Full downsampled UUV yaw in gray
    - ACA predictions on evaluated test windows in red
    - Calibrated interval around predictions
    - Vertical dashed lines on rising-edge fault triggers, or per-sample vertical markers
    """
    # Full UUV series (already downsampled in loader)
    t_all = np.asarray(data_loader.time_all, dtype=np.float32)
    y_all = np.asarray(data_loader.yaw_all, dtype=np.float32)

    # Trace values are in scaled space; convert back to original units
    pred_s = np.asarray(trace.get("prediction", []), dtype=np.float32)
    actual_s = np.asarray(trace.get("actual", []), dtype=np.float32)
    bound_s = np.asarray(trace.get("bound", []), dtype=np.float32)
    fail_flags = np.asarray(trace.get(fail_key, []), dtype=bool)

    if len(pred_s) == 0:
        return

    scale = float(getattr(data_loader.scaler, "scale_", np.array([1.0], dtype=np.float32))[0])
    pred_native = data_loader.scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
    actual_native = data_loader.scaler.inverse_transform(actual_s.reshape(-1, 1)).reshape(-1)
    band = bound_s * scale

    if getattr(data_loader, "use_differencing", False):
        prev = data_loader.test_prev_raw.detach().cpu().numpy().reshape(-1)
        m_native = min(len(prev), len(pred_native), len(actual_native), len(band))
        prev = prev[:m_native]
        pred_native = pred_native[:m_native]
        actual_native = actual_native[:m_native]
        band = band[:m_native]
        pred = prev + pred_native
        actual = prev + actual_native
    else:
        pred = pred_native
        actual = actual_native
    upper = pred + band
    lower = pred - band

    # Map test-window target timestamps back to global timeline
    start = int(data_loader.n_train_raw + data_loader.target_index_offset)
    if getattr(data_loader, "use_differencing", False):
        # dy_t target maps to raw y_{t+1}
        start += 1
    stop = min(start + len(pred), len(t_all))
    m = max(0, stop - start)
    if m == 0:
        return

    t_pred = t_all[start:stop]
    pred = pred[:m]
    actual = actual[:m]
    upper = upper[:m]
    lower = lower[:m]
    fail_flags = fail_flags[:m]

    # Optional persistence smoothing for plotted trigger markers
    persistent_fail = apply_fault_persistence(fail_flags, fault_persistence)
    trigger = np.zeros_like(persistent_fail, dtype=bool)
    if len(trigger) > 0:
        trigger[0] = bool(persistent_fail[0])
        trigger[1:] = persistent_fail[1:] & (~persistent_fail[:-1])

    plt.figure(figsize=(14, 5.2))
    plt.plot(t_all, y_all, color="gray", linewidth=1.0, label="Data from IMU (Yaw)")
    plt.plot(t_pred, pred, color="red", linewidth=1.0, alpha=0.9, label=prediction_label)
    plt.fill_between(
        t_pred,
        lower,
        upper,
        color="#8ecae6",
        alpha=0.35,
        label=f"Calibrated prediction interval (target coverage={(1.0 - alpha) * 100:.1f}%)",
    )

    if fail_marker_mode == "all":
        for tp in t_pred[persistent_fail]:
            plt.axvline(tp, color="royalblue", linestyle="--", linewidth=1.0, alpha=0.9)
    else:
        for tp in t_pred[trigger]:
            plt.axvline(tp, color="royalblue", linestyle="--", linewidth=1.0, alpha=0.9)

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Yaw Rate")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / output_filename, dpi=160)
    plt.close()


def run_inference(
    orchestrator: ACAOrchestrator,
    test_loader,
    max_samples: int = None,
    data_loader = None,
    fault_persistence: int = 1,
    label_overrides: Optional[np.ndarray] = None,
) -> dict:
    """
    Run inference on test data.
    
    Args:
        orchestrator: ACAOrchestrator instance
        test_loader: Test data loader
        max_samples: Optional limit on samples
        data_loader: Optional UUVDataLoader to get historical context
    """
    print("\nRunning inference...")
    
    # Re-finetuning dataset (for cloned temp engine in Tuner)
    # Use FULL training split as requested.
    history_window = None
    history_targets = None
    
    if data_loader:
        try:
            train_X, train_y = data_loader.get_train_data()
            history_window = train_X.to(orchestrator.engine.device)
            history_targets = train_y.to(orchestrator.engine.device)
            print(f"Initialized fine-tune set with FULL training split: {len(history_window)} samples")
        except Exception as e:
            print(f"Warning: Could not init history from train set: {e}")
    
    sample_count = 0
    trace = {
        "index": [],
        "prediction": [],
        "actual": [],
        "error": [],
        "bound": [],
        "uncertainty": [],
        "uncertainty_bound": [],
        "initial_fail": [],
        "raw_council_fail": [],
        "final_fail": [],
        "pred_label": [],
        "true_label": [],
    }
    for batch_x, batch_y in test_loader:
        for i in range(len(batch_x)):
            if max_samples and sample_count >= max_samples:
                break
            
            # Extract current sample tensors
            curr_window = batch_x[i:i+1].to(orchestrator.engine.device)
            curr_target = batch_y[i:i+1].to(orchestrator.engine.device)

            raw_window = None
            if data_loader is not None:
                try:
                    if getattr(data_loader, "use_differencing", False):
                        raw_np = data_loader.get_test_raw_window(sample_count)
                        if raw_np is not None:
                            raw_np = raw_np.reshape(1, -1, 1)
                            raw_window = torch.as_tensor(raw_np, dtype=torch.float32, device=orchestrator.engine.device)
                    else:
                        win_np = curr_window.detach().cpu().numpy().reshape(-1, 1)
                        raw_np = data_loader.scaler.inverse_transform(win_np).reshape(1, -1, 1)
                        raw_window = torch.as_tensor(raw_np, dtype=torch.float32, device=orchestrator.engine.device)
                except Exception:
                    raw_window = None
            
            result = orchestrator.process_sample(
                window=curr_window,
                actual=float(curr_target.item()),
                sample_id=sample_count,
                history_window=history_window,
                history_targets=history_targets,
                actual_label_override=(
                    None
                    if label_overrides is None or sample_count >= len(label_overrides)
                    else int(label_overrides[sample_count])
                ),
                raw_window=raw_window,
            )

            trace["index"].append(sample_count)
            trace["prediction"].append(float(result.case_file.prediction))
            trace["actual"].append(float(result.case_file.actual))
            trace["error"].append(float(result.case_file.error))
            trace["bound"].append(float(result.case_file.bound))
            trace["uncertainty"].append(float(result.case_file.uncertainty))
            ub = float("nan")
            if result.case_file.agent_results:
                sentry_payload = result.case_file.agent_results[0].payload or {}
                if "uncertainty_bound" in sentry_payload:
                    ub = float(sentry_payload["uncertainty_bound"])
            trace["uncertainty_bound"].append(ub)
            trace["initial_fail"].append(result.initial_decision.value == "FAIL")
            trace["raw_council_fail"].append(result.raw_final_decision.value == "FAIL")
            trace["final_fail"].append(result.final_decision.value == "FAIL")
            trace["pred_label"].append(1 if result.final_decision.value == "FAIL" else 0)
            if label_overrides is not None and sample_count < len(label_overrides):
                trace["true_label"].append(int(label_overrides[sample_count]))
            else:
                trace["true_label"].append(float("nan"))
            
            # Print progress every 50 samples
            if (sample_count + 1) % 50 == 0:
                metrics = orchestrator.get_metrics()
                print(f"  Processed {sample_count + 1} samples - "
                      f"F1: {metrics.f1_score:.4f}, "
                      f"Recall: {metrics.recall:.4f}")
            
            sample_count += 1
        
        if max_samples and sample_count >= max_samples:
            break
    
    print(f"\nProcessed {sample_count} total samples")
    persistent = apply_fault_persistence(np.asarray(trace["final_fail"], dtype=bool), fault_persistence)
    print(f"Persistent fail rate (k={fault_persistence}): {float(persistent.mean()):.3f}")
    return trace


def write_plot_run_log(
    out_dir: Path,
    args,
    config: ACAConfig,
    orchestrator: ACAOrchestrator,
    run_log_dir: Path,
) -> None:
    """Write a concise run summary inside the timestamped run log directory."""
    metrics = orchestrator.get_metrics()
    summary_path = run_log_dir / "run_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ACA Run Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"uuv_file: {args.uuv_file}\n")
        f.write(f"uuv_downsample: {args.uuv_downsample}\n")
        f.write(f"uuv_train_frac: {args.uuv_train_frac}\n")
        f.write(f"uuv_use_differencing: {args.uuv_use_differencing}\n")
        f.write("\n")
        f.write("Model / Training Hyperparameters\n")
        f.write("-" * 60 + "\n")
        f.write(f"window_size: {config.window_size}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"lstm_hidden_dim: {config.lstm_hidden_dim}\n")
        f.write(f"lstm_num_layers: {config.lstm_num_layers}\n")
        f.write(f"lstm_dropout: {config.lstm_dropout}\n")
        f.write(f"mc_samples: {config.mc_samples}\n")
        f.write(f"fine_tune_lr: {config.fine_tune_lr}\n")
        f.write("\n")
        f.write("Detection / Council\n")
        f.write("-" * 60 + "\n")
        f.write(f"fault_persistence: {args.fault_persistence}\n")
        f.write(f"council_confirm_k: {config.council_confirm_k}\n")
        f.write(f"council_release_k: {config.council_release_k}\n")
        f.write(f"council_temporal_filter_enabled: {config.ENABLE_COUNCIL_TEMPORAL_FILTER}\n")
        f.write(f"sentry_uncertainty_only_fail: {config.sentry_uncertainty_only_fail}\n")
        f.write("\n")
        f.write("Metrics\n")
        f.write("-" * 60 + "\n")
        f.write(f"total: {metrics.total}\n")
        f.write(f"tp: {metrics.true_positives}\n")
        f.write(f"fp: {metrics.false_positives}\n")
        f.write(f"tn: {metrics.true_negatives}\n")
        f.write(f"fn: {metrics.false_negatives}\n")
        f.write(f"precision: {metrics.precision:.6f}\n")
        f.write(f"recall: {metrics.recall:.6f}\n")
        f.write(f"f1: {metrics.f1_score:.6f}\n")
        f.write(f"false_positive_rate: {metrics.false_positive_rate:.6f}\n")
        f.write(f"override_rate: {metrics.override_rate:.6f}\n")
        f.write("\n")
        f.write("Paths\n")
        f.write("-" * 60 + "\n")
        f.write(f"plot_dir: {out_dir.resolve()}\n")
        f.write(f"terminal_log_dir: {run_log_dir.resolve()}\n")

        # --- V&V Summary ---
        f.write("\n")
        f.write("V&V Summary (Agentic System Verification & Validation)\n")
        f.write("=" * 60 + "\n")
        try:
            vv = orchestrator.get_vv_summary()

            # Requirements Engineer (Normal Mode V&V)
            re = vv.get('requirements_engineer', {})
            f.write("\n[Requirements Engineer -- Normal Mode V&V]\n")
            f.write(f"  Num of agent calls : {re.get('total_calls', 0)}\n")
            f.write(f"  Num of FAIL  : {re.get('fail_count', 0)}\n")
            f.write(f"  Test result : {re.get('test_result', 'N/A')}\n")
            re_fails = re.get('fail_samples', [])
            if re_fails:
                f.write("  Violation details (up to 5):\n")
                for s in re_fails:
                    req_sec = s.get('requirement_section') or 'N/A'
                    reason = (s.get('veto_reason') or s.get('reasoning') or 'N/A')[:180].rstrip()
                    f.write(f"    - Sample {s['sample_id']:>4d} | {req_sec} | {reason}\n")

            # Failure Manager (Failure Mode V&V)
            fm = vv.get('failure_manager', {})
            f.write("\n[Failure Manager -- Failure Mode V&V]\n")
            f.write(f"  Num of agent calls : {fm.get('total_calls', 0)}\n")
            f.write(f"  Num of FAIL  : {fm.get('fail_count', 0)}\n")
            f.write(f"  Test result : {fm.get('test_result', 'N/A')}\n")
            fm_fails = fm.get('fail_samples', [])
            if fm_fails:
                f.write("  Violation details (up to 5):\n")
                for s in fm_fails:
                    peak = s.get('peak_deviation')
                    converging = s.get('is_converging')
                    osc = s.get('oscillation_count')
                    assess = (s.get('fm_assessment') or s.get('reasoning') or 'N/A')[:160].rstrip()
                    conv_str = "CONVERGING" if converging else ("DIVERGING" if converging is not None else "N/A")
                    peak_str = f"{peak:.2f}" if peak is not None else "N/A"
                    osc_str = str(osc) if osc is not None else "N/A"
                    f.write(
                        f"    - Sample {s['sample_id']:>4d} | "
                        f"peak={peak_str} | {conv_str} | osc={osc_str} | {assess}\n"
                    )

            # System Engineer (Active Optimizer V&V)
            se = vv.get('system_engineer', {})
            f.write("\n[System Engineer -- Active Optimizer]\n")
            f.write(f"  Num of agent calls : {se.get('total_calls', 0)}\n")
            f.write(f"  Fail Votes  : {se.get('fail_count', 0)}\n")
            se_proposals = se.get('tuning_proposals', [])
            if se_proposals:
                f.write(f"  Gain-Tuning Proposals ({len(se_proposals)} unique samples, triggered by FM/RE FAIL, showing up to 5):\n")
                for s in se_proposals[:5]:
                    vote = s.get('vote', 'N/A')
                    triggered = s.get('triggered_by', '?')
                    reason = (s.get('tuning_reasoning') or s.get('reasoning') or 'N/A')[:180].rstrip()
                    proposal = s.get('tuning_proposal', {})
                    f.write(f"    - Sample {s['sample_id']:>4d} | Triggered by: {triggered} | SE Vote={vote} | Params: {proposal}\n")
                    f.write(f"      Reason: {reason}\n")
            else:
                f.write("  No gain-tuning proposals generated (FM and RE both PASS on all samples).\n")
        except Exception as e:
            f.write(f"  (V&V summary unavailable: {e})\n")
        f.write("\n")



def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AIVV for anomaly detection"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='uuv',
        choices=['uuv'],
        help='Dataset to run (UUV only)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--uuv-file',
        type=str,
        default=str(Path(__file__).resolve().parent / 'data' / 'uuv' / 'UUV_yaw_fault_0_new.txt'),
        help='Path to UUV yaw file (for --dataset uuv)'
    )
    parser.add_argument(
        '--uuv-downsample',
        type=int,
        default=20,
        help='Downsample factor for UUV data'
    )
    parser.add_argument(
        '--uuv-train-frac',
        type=float,
        default=0.7,
        help='Train fraction for UUV split'
    )
    parser.add_argument(
        '--forecast-horizon',
        type=int,
        default=1,
        help='Forecast horizon for next-step target creation'
    )
    parser.add_argument(
        '--uuv-use-differencing',
        action='store_true',
        help='Train/predict on delta yaw (dy_t) instead of absolute yaw'
    )
    parser.add_argument(
        '--uuv-fault-start-index',
        type=int,
        default=None,
        help='Explicit start index of the simulated UUV fault in the downsampled series'
    )
    parser.add_argument(
        '--uuv-fault-end-index',
        type=int,
        default=None,
        help='Explicit end index of the simulated UUV fault in the downsampled series'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=None,
        help='Override config window size'
    )
    parser.add_argument(
        '--lstm-num-layers',
        type=int,
        default=None,
        help='Override LSTM number of layers'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=None,
        help='Override LSTM hidden dimension'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='Override LSTM/MC dropout probability'
    )
    parser.add_argument(
        '--mc-samples',
        type=int,
        default=None,
        help='Override MC dropout sample count at inference'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override fine-tuning learning rate (used by model training/fine-tune)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--fault-persistence',
        type=int,
        default=3,
        help='Minimum consecutive FAIL steps to keep fault active in tracking plots'
    )
    parser.add_argument(
        '--council-confirm-k',
        type=int,
        default=2,
        help='Council temporal filter: consecutive FAILs required to enter failure mode'
    )
    parser.add_argument(
        '--council-release-k',
        type=int,
        default=2,
        help='Council temporal filter: consecutive PASSes required to exit failure mode'
    )
    parser.add_argument(
        '--disable-council-temporal-filter',
        action='store_true',
        help='Disable council temporal debouncing (uses raw council decision each step)'
    )
    parser.add_argument(
        '--sentry-uncertainty-only-fail',
        action='store_true',
        help='If set, Sentry can trigger FAIL on high uncertainty even when conformal bound is not exceeded'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Save inference tracking plots'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (overrides config.SEED if provided)'
    )
    parser.add_argument(
        '--run-log-dir',
        type=str,
        default=None,
        help='Optional explicit log directory for this run (default: logs/run_<timestamp>)'
    )
    
    args = parser.parse_args()

    run_log_dir = setup_run_terminal_logging(
        run_dir=Path(args.run_log_dir) if args.run_log_dir else None
    )
    print(f"Run logs directory: {run_log_dir.resolve()}")
    
    # Setup
    setup_logging(args.verbose)
    config = ACAConfig()
    
    # Set seed (command-line overrides config)
    if args.seed is not None:
        config.SEED = args.seed
    set_seed(config.SEED)
    
    if args.window_size is not None:
        config.window_size = int(args.window_size)
    if args.lstm_num_layers is not None:
        config.lstm_num_layers = int(args.lstm_num_layers)
    if args.hidden_dim is not None:
        config.lstm_hidden_dim = int(args.hidden_dim)
    if args.dropout is not None:
        config.lstm_dropout = float(args.dropout)
    if args.mc_samples is not None:
        config.mc_samples = int(args.mc_samples)
    if args.lr is not None:
        config.fine_tune_lr = float(args.lr)

    # Domain mode for agent prompts/rules (UUV-only)
    config.domain = 'uuv'
    config.council_confirm_k = int(args.council_confirm_k)
    config.council_release_k = int(args.council_release_k)
    config.ENABLE_COUNCIL_TEMPORAL_FILTER = not bool(args.disable_council_temporal_filter)
    config.sentry_uncertainty_only_fail = bool(args.sentry_uncertainty_only_fail)
    
    # Resolve runtime device (supports mps/cuda/xpu/cpu)
    device = config.get_device()
    print(f"Using device: {device}")
    
    print("=" * 60)
    print("AIVV")
    print("Anomaly Detection with Anti-Forgetting")
    print("=" * 60)
    print(
        "Hyperparams: "
        f"layers={config.lstm_num_layers}, hidden={config.lstm_hidden_dim}, "
        f"dropout={config.lstm_dropout}, mc_samples={config.mc_samples}, "
        f"lr={config.fine_tune_lr}, window={config.window_size}, epochs={args.epochs}"
    )
    
    # Load data with train/test split
    print("\nLoading data...")
    label_overrides = None
    config.lstm_input_dim = 1
    data_loader = UUVDataLoader(
        file_path=args.uuv_file,
        window_size=config.window_size,
        train_ratio=args.uuv_train_frac,
        downsample=args.uuv_downsample,
        horizon=args.forecast_horizon,
        use_differencing=args.uuv_use_differencing,
        device=config.get_device(),
    )
    fault_window = resolve_uuv_fault_window(
        file_path=args.uuv_file,
        start_index=args.uuv_fault_start_index,
        end_index=args.uuv_fault_end_index,
    )
    if fault_window is not None:
        label_overrides = data_loader.build_test_fault_labels(*fault_window)
        positive_count = int(np.sum(label_overrides))
        print(
            "Using explicit UUV fault labels: "
            f"global indices {fault_window[0]}-{fault_window[1]} "
            f"-> {positive_count} labeled test windows"
        )
    else:
        print("No explicit UUV fault labels configured; evaluation metrics will be skipped.")
    
    print(f"Dataset stats: {data_loader.get_stats()}")
    
    # Initialize engine
    print("\nInitializing engine...")
    engine = UUVEngine(config=config)
    
    # Train with proper calibration
    train_X, train_y = data_loader.get_train_data()
    n_cal = max(1, int(0.2 * len(train_X)))
    cal_X, cal_y = train_X[-n_cal:], train_y[-n_cal:]
    fit_X, fit_y = train_X[:-n_cal], train_y[:-n_cal]

    train_loader = DataLoader(
        TensorDataset(fit_X, fit_y),
        batch_size=32,
        shuffle=True
    )
    cal_data = (cal_X, cal_y)
    train_engine(engine, train_loader, epochs=args.epochs, cal_data=cal_data)
    engine.register_training_reference(
        train_windows=fit_X,
        train_targets=fit_y,
        calibration_windows=cal_X,
        calibration_targets=cal_y,
        raw_train_series=getattr(data_loader, 'x_train_raw', None),
        dataset_stats=data_loader.get_stats(),
        scaler_stats={
            'mean': [float(v) for v in np.asarray(getattr(data_loader.scaler, 'mean_', np.asarray([], dtype=np.float32))).reshape(-1)],
            'scale': [float(v) for v in np.asarray(getattr(data_loader.scaler, 'scale_', np.asarray([], dtype=np.float32))).reshape(-1)],
        },
    )
    
    # Create orchestrator
    orchestrator = ACAOrchestrator(
        config=config,
        engine=engine,
        run_log_dir=str(run_log_dir),
    )
    
    # Run inference
    test_loader = data_loader.get_test_loader()
    trace = run_inference(
        orchestrator,
        test_loader,
        max_samples=args.samples,
        data_loader=data_loader,
        fault_persistence=args.fault_persistence,
        label_overrides=label_overrides,
    )

    if args.plot:
        out_dir = run_log_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_inference_plots(trace, out_dir=out_dir, fault_persistence=args.fault_persistence)
        if args.dataset == 'uuv':
            save_uuv_fdi_style_plot(
                trace,
                data_loader=data_loader,
                out_dir=out_dir,
                alpha=config.default_alpha,
                fault_persistence=args.fault_persistence,
            )
            save_uuv_fdi_style_plot(
                trace,
                data_loader=data_loader,
                out_dir=out_dir,
                alpha=config.default_alpha,
                fault_persistence=args.fault_persistence,
                fail_key="initial_fail",
                output_filename="uuv_no_agent_fdi_style.png",
                prediction_label="Predicted Yaw (no-agent)",
                title="FDI of Yaw Rate (no-agent)",
                fail_marker_mode="all",
            )
        write_plot_run_log(
            out_dir=out_dir,
            args=args,
            config=config,
            orchestrator=orchestrator,
            run_log_dir=run_log_dir,
        )
        print(f"Saved tracking plots to: {out_dir.resolve()}")
    
    # Print results
    print("\n" + orchestrator.print_metrics_report())


if __name__ == "__main__":
    main()
