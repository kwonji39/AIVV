#!/usr/bin/env python3
"""
Plot regeneration script supporting both single plots and combined 3x3 figures.

This script regenerates plots from ACA run results. It can:
1. Generate a single plot from one run directory (--run-dir)
2. Generate a combined 3x3 subplot figure from 9 run directories (--run-dirs)
3. Use version-aware plotting logic for different job versions ("general", "math_only", "no_inspector_tuner")

Usage:
    python regenerate_plots_batch.py --run-dir <path> [--version general]
    python regenerate_plots_batch.py --run-dirs <path1> <path2> ... <path9> [--versions v1 v2 ... v9]
    python regenerate_plots_batch.py --default-jobs
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION: Default 9 jobs (used by --default-jobs flag)
# ============================================================================

DEFAULT_JOBS = [
    {"label": "(a) Baseline", "version": "math_only", "run_dir": "logs/run_batch_20260321_215630/seed_9_run10/math_sentry_only", "prefix": "math_sentry_only_aca_"},
    {"label": "(b) Baseline", "version": "math_only", "run_dir": "logs/run_batch_20260321_213531/seed_21_run22/math_sentry_only", "prefix": "math_sentry_only_aca_"},
    {"label": "(c) Baseline", "version": "math_only", "run_dir": "logs/run_batch_20260321_201433/seed_71_run52/math_sentry_only", "prefix": "math_sentry_only_aca_"},
    {"label": "(d) + Council", "version": "no_inspector_tuner", "run_dir": "logs/run_batch_20260330_105406/seed_16_run1", "prefix": "aca_"},
    {"label": "(e) + Council", "version": "no_inspector_tuner", "run_dir": "logs/run_batch_20260330_030453/seed_19_run1", "prefix": "aca_"},
    {"label": "(f) + Council", "version": "no_inspector_tuner", "run_dir": "logs/run_batch_20260330_030727/seed_30_run1", "prefix": "aca_"},
    {"label": "(g) AIVV", "version": "general", "run_dir": "logs/data_0_new/run_batch_20260318_144715/seed_9_run10", "prefix": "aca_"},
    {"label": "(h) AIVV", "version": "general", "run_dir": "logs/data_1/run_batch_20260319_154445/seed_21_run2", "prefix": "aca_"},
    {"label": "(i) AIVV", "version": "general", "run_dir": "logs/data_2/run_batch_20260320_170213/seed_71_run9", "prefix": "aca_"},
]

# ============================================================================
# CORE PLOTTING FUNCTIONS
# ============================================================================


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


def cumulative_prf(pred_labels: np.ndarray, true_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def extract_trace_from_json_general(json_data: dict, run_summary: dict = None) -> dict:
    """
    Extract trace information from ACA run JSON file (for "general" and "math_only" versions).
    Used by regenerate_plots.py logic.
    """
    sample_logs = json_data.get('sample_logs', [])
    
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
    
    for sample in sample_logs:
        sample_id = sample.get('sample_id', 0)
        trace["index"].append(sample_id)
        
        # Extract predictions and bounds from sentry (first agent)
        agents = sample.get('agent_decisions', [])
        
        if agents:
            agent = agents[0]  # Get sentry decision
            payload = agent.get('payload_summary', {})
            
            prediction = payload.get('prediction', 0.0)
            error = payload.get('error', 0.0)
            bound = payload.get('bound', 0.0)
            
            # Reconstruct actual from error
            actual = prediction - error if error > 0 else prediction
            
            trace["prediction"].append(float(prediction))
            trace["actual"].append(float(actual))
            trace["error"].append(float(error))
            trace["bound"].append(float(bound))
            
            # Uncertainty data
            trace["uncertainty"].append(float(payload.get('uncertainty', 0.0)))
            trace["uncertainty_bound"].append(float(payload.get('uncertainty_bound', 0.0)))
            
            # Decision flags
            agent_decision = agent.get('decision', 'PASS')
            initial_fail = agent_decision == 'FAIL'
            trace["initial_fail"].append(initial_fail)
            
        # Final decision
        final_decision = sample.get('final_decision', 'PASS')
        final_fail = final_decision == 'FAIL'
        trace["final_fail"].append(final_fail)
        
        # Raw council fail
        trace["raw_council_fail"].append(False)
        
        # True label (ground truth)
        true_label = sample.get('actual_label', 0)
        trace["true_label"].append(float(true_label))
        
        # Prediction label
        pred_label = 1 if final_fail else 0
        trace["pred_label"].append(pred_label)
    
    # Convert to numpy arrays
    for key in trace:
        trace[key] = np.array(trace[key])
    
    return trace


def parse_trace_json_ablation(trace_file: Path) -> Dict:
    """
    Parse trace data from JSON file (for "no_inspector_tuner" version).
    Used by regenerate_ablation_plots.py logic.
    """
    try:
        with open(trace_file, 'r') as f:
            trace_json = json.load(f)
    except Exception as e:
        print(f"Error reading {trace_file}: {e}")
        return {}

    trace = {}
    for key, values in trace_json.items():
        if key == "true_label":
            # Handle NaN values in true_label
            trace[key] = np.array([v if v is not None else np.nan for v in values], dtype=np.float32)
        elif key in ["initial_fail", "raw_council_fail", "final_fail"]:
            trace[key] = np.array(values, dtype=bool)
        elif key in ["pred_label"]:
            trace[key] = np.array(values, dtype=np.int32)
        else:
            trace[key] = np.array(values, dtype=np.float32)
    
    return trace


def load_uuv_data_raw_and_downsampled(uuv_file: str, downsample: int = 20) -> tuple:
    """
    Load UUV data and return both raw and downsampled versions.
    
    Returns: (time_raw, yaw_raw, time_downsampled, yaw_downsampled)
    """
    try:
        data = np.loadtxt(uuv_file, skiprows=1)
        yaw_raw = data[:, 1]
        time_raw = np.arange(len(yaw_raw), dtype=np.float32)  # index-based, not timestamp
        
        # Downsample
        if downsample > 1:
            yaw_downsampled = yaw_raw[::downsample]
            time_downsampled = np.arange(len(yaw_downsampled), dtype=np.float32)  # 0, 1, 2, ..., ~1500
        else:
            time_downsampled = time_raw
            yaw_downsampled = yaw_raw
        
        return (np.asarray(time_raw, dtype=np.float32), 
                np.asarray(yaw_raw, dtype=np.float32),
                np.asarray(time_downsampled, dtype=np.float32), 
                np.asarray(yaw_downsampled, dtype=np.float32))
    except Exception as e:
        print(f"  Warning: Could not load UUV data from {uuv_file}: {e}")
        return None, None, None, None


def load_trace_data_for_version(run_dir: Path, version: str = "general", prefix: str = None) -> Optional[dict]:
    """
    Load trace data from run directory, dispatching based on version.
    
    Args:
        run_dir: Path to run directory
        version: "general", "math_only", or "no_inspector_tuner"
        prefix: Prefix for identifying which data to use (e.g., "math_sentry_only_aca_")
        
    Returns:
        Trace dictionary or None
    """
    run_dir = Path(run_dir)
    
    if not run_dir.exists():
        print(f"  Warning: Run directory not found: {run_dir}")
        return None
    
    if version in ["general", "math_only"]:
        # Load from aca_run_*.json files (regenerate_plots.py logic)
        json_files = list(run_dir.glob("aca_run_*.json"))
        if not json_files:
            print(f"  Warning: No aca_run_*.json file found in {run_dir}")
            return None
        
        json_file = json_files[0]
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            # Try to load run_summary.txt for additional config
            summary_file = run_dir / "run_summary.txt"
            run_summary = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    run_summary = f.read()
            
            trace = extract_trace_from_json_general(json_data, run_summary)
            return trace
        except Exception as e:
            print(f"  Warning: Error loading trace data: {e}")
            return None
    
    elif version == "no_inspector_tuner":
        # Load from trace.json in mode/plots/ directory (regenerate_ablation_plots.py logic)
        possible_paths = [
            run_dir / "no_inspector_tuner" / "plots" / "trace.json",
            run_dir / "no_inspector_tuner" / "trace.json",
            run_dir / "plots" / "trace.json",
            run_dir / "trace.json",
        ]
        
        trace_file = None
        for path in possible_paths:
            if path.exists():
                trace_file = path
                break
        
        if trace_file is None:
            print(f"  Warning: trace.json not found in {run_dir} or subdirectories")
            return None
        
        try:
            trace = parse_trace_json_ablation(trace_file)
            return trace
        except Exception as e:
            print(f"  Warning: Error loading trace data: {e}")
            return None
    
    else:
        print(f"  Warning: Unknown version: {version}")
        return None


def plot_fdi_style_for_version(
    ax,
    run_dir: Path,
    version: str = "general",
    prefix: str = "aca_",
    fault_persistence: int = 1,
    label_letter: Optional[str] = None,
) -> bool:
    """
    Plot FDI-style plot on a single axis using version-specific logic.
    
    Args:
        ax: matplotlib axis to draw on
        run_dir: Path to run directory
        version: "general", "math_only", or "no_inspector_tuner"
        prefix: Prefix for identifying which data to use
        fault_persistence: Fault persistence parameter
        label_letter: Letter label ("(a)", "(b)", etc.) to place in top-left
        
    Returns:
        True if successful, False otherwise
    """
    run_dir = Path(run_dir)
    
    # Load trace data
    trace = load_trace_data_for_version(run_dir, version, prefix)
    if trace is None or len(trace.get('index', [])) == 0:
        ax.text(0.5, 0.5, 'Failed to load data', ha='center', va='center', transform=ax.transAxes)
        return False
    
    if version in ["general", "math_only"]:
        # Use regenerate_plots.py logic
        summary_file = run_dir / "run_summary.txt"
        
        # Parse run summary to extract UUV file path and parameters
        uuv_file = None
        downsample = 20
        train_frac = 0.7
        window_size = 10
        horizon = 1
        use_differencing = False
        
        # Fall back to ablation.txt if run_summary.txt does not exist
        # if not summary_file.exists():
        #     summary_file = run_dir / "ablation.txt"
            
        if not summary_file.exists():
            # Check run_dir first, then parent directory
            if (run_dir / "ablation.txt").exists():
                summary_file = run_dir / "ablation.txt"
            elif (run_dir.parent / "ablation.txt").exists():
                summary_file = run_dir.parent / "ablation.txt"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                for line in f:
                    if 'uuv_file:' in line:
                        uuv_file = line.split(':', 1)[1].strip()
                    elif 'uuv_downsample:' in line:
                        downsample = int(line.split(':', 1)[1].strip())
                    elif 'uuv_train_frac:' in line:
                        train_frac = float(line.split(':', 1)[1].strip())
                    elif 'window_size:' in line:
                        window_size = int(line.split(':', 1)[1].strip())
                    elif 'uuv_use_differencing:' in line:
                        use_differencing = line.split(':', 1)[1].strip().lower() == 'true'
        
        # If we couldn't find uuv_file in summary, use default
        if uuv_file is None:
            print("not found uuv_file in summary, using default")
            uuv_file = "data/uuv/UUV_yaw_fault_0_new.txt"
        
        # Construct full path to UUV file
        uuv_path = Path(__file__).parent / uuv_file
        if not uuv_path.exists():
            ax.text(0.5, 0.5, 'UUV not found', ha='center', va='center', transform=ax.transAxes)
            return False
        
        # Load UUV data
        time_raw, yaw_raw, time_all, yaw_all = load_uuv_data_raw_and_downsampled(str(uuv_path), downsample=downsample)
        if time_all is None or yaw_all is None:
            return False
        
        # Get trace data
        pred = np.asarray(trace.get("prediction", []), dtype=np.float32)
        bound = np.asarray(trace.get("bound", []), dtype=np.float32)
        final_fail = np.asarray(trace.get("final_fail", []), dtype=bool)
        
        if len(pred) == 0 or len(time_all) == 0:
            return False
        
        # Compute scaler statistics from full UUV data
        mean = np.mean(yaw_all)
        std = np.std(yaw_all)
        if std == 0:
            std = 1.0
        scale = std
        
        # Inverse transform: unscale predictions and bounds
        pred_native = pred * scale + mean
        band = bound * scale
        upper = pred_native + band
        lower = pred_native - band
        
        # Compute exact test start position
        n_train_raw = int(len(yaw_raw) * train_frac)
        target_index_offset = int(window_size + horizon - 1)
        start_raw = int(n_train_raw + target_index_offset)
        if use_differencing:
            start_raw += 1
        
        # Convert to downsampled space
        start = start_raw // downsample
        stop = min(start + len(pred), len(time_all))
        m = max(0, stop - start)
        
        if m == 0:
            return False
        
        pred_native = pred_native[:m]
        upper = upper[:m]
        lower = lower[:m]
        final_fail = final_fail[:m]
        
        # Apply fault persistence for trigger detection
        persistent_fail = apply_fault_persistence(final_fail, fault_persistence)
        trigger = np.zeros_like(persistent_fail, dtype=bool)
        if len(trigger) > 0:
            trigger[0] = bool(persistent_fail[0])
            trigger[1:] = persistent_fail[1:] & (~persistent_fail[:-1])
        
        # Use actual time values for x-axis (not indices)
        x_all = time_all
        t_pred_idx = time_all[start:start + m]
        
        # Plot
        ax.plot(x_all, yaw_all, color="gray", linewidth=1.0)
        ax.plot(t_pred_idx, pred_native, color="red", linewidth=1.0, alpha=0.9)
        ax.fill_between(
            t_pred_idx,
            lower,
            upper,
            color="#8ecae6",
            alpha=0.35,
        )
        
        # Mark faults based on prefix
        if "math_sentry_only" in prefix:
            # math_sentry_only: Mark ALL persistent failures with thick dashed lines
            for tp in t_pred_idx[persistent_fail]:
                ax.axvline(tp+8, color="royalblue", linestyle="--", linewidth=1.0, alpha=0.9)
        else:
            # Other modes: Mark only rising-edge triggers
            for tp in t_pred_idx[trigger]:
                ax.axvline(tp+8, color="royalblue", linestyle="--", linewidth=1.0, alpha=0.9)
        
        ax.set_xlabel("Time Step", fontsize=13, fontweight='bold')
        ax.set_ylabel("Yaw angle (deg)", fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)
        
    elif version == "no_inspector_tuner":
        # Dynamically find uuv_file from ablation.txt or run_summary.txt
        uuv_file_str = None
        summary_file = run_dir / "run_summary.txt"
        if not summary_file.exists():
            if (run_dir / "ablation.txt").exists():
                summary_file = run_dir / "ablation.txt"
            elif (run_dir.parent / "ablation.txt").exists():
                summary_file = run_dir.parent / "ablation.txt"

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                for line in f:
                    if 'uuv_file:' in line:
                        uuv_file_str = line.split(':', 1)[1].strip()
                        break

        if uuv_file_str is None:
            print("not found uuv_file in summary, using default")
            uuv_file_str = "data/uuv/UUV_yaw_fault_0_new.txt"

        uuv_file = Path(__file__).parent / uuv_file_str
        if not uuv_file.exists():
            ax.text(0.5, 0.5, 'UUV not found', ha='center', va='center', transform=ax.transAxes)
            return False
        
        # Load UUV data
        time_raw, yaw_raw, time_all, yaw_all = load_uuv_data_raw_and_downsampled(str(uuv_file), downsample=20)
        if time_all is None or yaw_all is None:
            return False
        
        # Get trace data
        pred = np.asarray(trace.get("prediction", []), dtype=np.float32)
        bound = np.asarray(trace.get("bound", []), dtype=np.float32)
        final_fail = np.asarray(trace.get("final_fail", []), dtype=bool)
        
        if len(pred) == 0 or len(time_all) == 0:
            return False
        
        # Compute scaler statistics from full UUV data
        mean = np.mean(yaw_all)
        std = np.std(yaw_all)
        if std == 0:
            std = 1.0
        scale = std
        
        # Inverse transform: unscale predictions and bounds
        pred_native = pred * scale + mean
        band = bound * scale
        upper = pred_native + band
        lower = pred_native - band
        
        # Compute test start position (hardcoded from ablation.py)
        n_total_raw = len(yaw_raw)
        train_frac = 0.7
        window_size = 10
        horizon = 2
        use_differencing = False
        downsample = 20
        
        n_train_raw = int(n_total_raw * train_frac)
        target_index_offset = int(window_size + horizon - 1)
        start_raw = int(n_train_raw + target_index_offset)
        if use_differencing:
            start_raw += 1
        
        start = start_raw // downsample
        stop = min(start + len(pred), len(time_all))
        m = max(0, stop - start)
        
        if m == 0:
            return False
        
        pred_native = pred_native[:m]
        upper = upper[:m]
        lower = lower[:m]
        final_fail = final_fail[:m]
        
        # Apply fault persistence
        persistent_fail = apply_fault_persistence(final_fail, fault_persistence)
        trigger = np.zeros_like(persistent_fail, dtype=bool)
        if len(trigger) > 0:
            trigger[0] = bool(persistent_fail[0])
            trigger[1:] = persistent_fail[1:] & (~persistent_fail[:-1])
        
        # Use actual time values for x-axis (not indices)
        x_all = time_all
        t_pred_idx = time_all[start:start + m]
        
        # Plot
        ax.plot(x_all, yaw_all, color="gray", linewidth=1.0)
        ax.plot(t_pred_idx, pred_native, color="red", linewidth=1.0, alpha=0.9)
        ax.fill_between(
            t_pred_idx,
            lower,
            upper,
            color="#8ecae6",
            alpha=0.35,
        )
        
        # Mark ALL persistent failures (matching ablation.py behavior)
        for tp in t_pred_idx[persistent_fail]:
            ax.axvline(tp+8, color="royalblue", linestyle="--", linewidth=1.0, alpha=0.9)
        
        ax.set_xlabel("Time Step", fontsize=13, fontweight='bold')
        ax.set_ylabel("Yaw angle (deg)", fontsize=13, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)
    
    else:
        ax.text(0.5, 0.5, 'Unknown version', ha='center', va='center', transform=ax.transAxes)
        return False
    
    # Add letter label in top-left corner
    if label_letter:
        ax.text(
            0.02, 0.98,
            label_letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        # ax.text(
        #     0.5, -0.15,          # x=가운데, y=axes 아래 밖
        #     label_letter,
        #     transform=ax.transAxes,
        #     fontsize=12,
        #     fontweight='bold',
        #     horizontalalignment='center',
        #     verticalalignment='top',
        #     # bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        # )
    
    return True





def plot_single_detailed(
    run_dir: Path,
    version: str = "general",
    prefix: str = "aca_",
    fault_persistence: int = 1,
) -> bool:
    """
    Generate single detailed plot (used for --run-dir mode).
    Creates plots using version-specific logic.
    
    Args:
        run_dir: Path to run directory
        version: "general", "math_only", or "no_inspector_tuner"
        prefix: Filename prefix
        fault_persistence: Fault persistence parameter
        
    Returns:
        True if successful, False otherwise
    """
    run_dir = Path(run_dir)
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return False
    
    # Load trace data using version-aware loader
    trace = load_trace_data_for_version(run_dir, version, prefix)
    if trace is None:
        print(f"Error: Failed to load trace data from {run_dir}")
        return False
    
    print(f"Loaded {len(trace.get('index', []))} samples")
    
    # Extract arrays
    t = np.asarray(trace.get("index", []), dtype=np.int32)
    pred = np.asarray(trace.get("prediction", []), dtype=np.float32)
    actual = np.asarray(trace.get("actual", []), dtype=np.float32)
    err = np.asarray(trace.get("error", []), dtype=np.float32)
    bound = np.asarray(trace.get("bound", []), dtype=np.float32)
    final_fail = np.asarray(trace.get("final_fail", []), dtype=bool)
    pred_label = np.asarray(trace.get("pred_label", []), dtype=np.int32)
    true_label = np.asarray(trace.get("true_label", []), dtype=np.float32)
    
    persistent_fail = apply_fault_persistence(final_fail, fault_persistence)
    
    # Create output directory
    out_dir = run_dir / "plots_regenerated"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # 1) Prediction vs actual
    print("  - Generating prediction vs actual plot...")
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
    plt.savefig(out_dir / f"{prefix}pred_vs_actual.png", dpi=100)
    plt.close()
    
    # 2) Error vs conformal bound
    print("  - Generating residual vs bound plot...")
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
    plt.savefig(out_dir / f"{prefix}residual_vs_bound.png", dpi=100)
    plt.close()
    
    # 2b) Uncertainty vs calibrated uncertainty bound (if available)
    print("  - Generating uncertainty vs bound plot...")
    uncertainty = np.asarray(trace.get("uncertainty", []), dtype=np.float32)
    uncertainty_bound = np.asarray(trace.get("uncertainty_bound", []), dtype=np.float32)
    
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
        plt.savefig(out_dir / f"{prefix}uncertainty_vs_bound.png", dpi=160)
        plt.close()
    
    # 3) Decision timeline
    print("  - Generating decision timeline plot...")
    init_fail = np.asarray(trace.get("initial_fail", []), dtype=bool)
    raw_council_fail = np.asarray(trace.get("raw_council_fail", []), dtype=bool)
    
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
    plt.savefig(out_dir / f"{prefix}decision_timeline.png", dpi=100)
    plt.close()
    
    # 4) Cumulative PRF1
    print("  - Generating cumulative PRF1 plot...")
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
        plt.savefig(out_dir / f"{prefix}cumulative_prf1.png", dpi=100)
        plt.close()
    
    print(f"✓ All plots saved to {out_dir}")
    return True


# ============================================================================
# COMBINED 3X3 FIGURE GENERATION
# ============================================================================


def create_combined_figure(
    jobs: List[Dict],
    output_file: str = "combined_plot.png",
    fault_persistence: int = 1,
) -> bool:
    """
    Create a 3x3 combined figure from 9 jobs with FDI-style plots only.
    
    Args:
        jobs: List of 9 job dictionaries, each with:
              - "label": display label (e.g., "(a) Baseline A")
              - "version": plot version ("general", "math_only", or "no_inspector_tuner")
              - "run_dir": path to run directory
              - "prefix": optional filename prefix
        output_file: Output filename
        fault_persistence: Fault persistence parameter
        
    Returns:
        True if successful, False otherwise
    """
    if len(jobs) != 9:
        print(f"Error: Expected 9 jobs, got {len(jobs)}")
        return False
    
    print(f"\nCreating 3x3 combined figure with FDI-style plots from {len(jobs)} runs...")
    print(f"Output file: {output_file}")
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 8))
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Plot each run with FDI-style plots
    for idx, job in enumerate(jobs):
        ax = axes_flat[idx]
        row = idx // 3
        col = idx % 3
        
        label = job.get("label", f"({chr(97 + idx)})")
        version = job.get("version", "general")
        run_dir = Path(job.get("run_dir", ""))
        prefix = job.get("prefix", "aca_")
        
        print(f"  [{idx + 1}/9] FDI-plot {label}: {run_dir} (version={version})")
        
        success = plot_fdi_style_for_version(
            ax,
            run_dir,
            version=version,
            prefix=prefix,
            fault_persistence=fault_persistence,
            label_letter=label,
        )
        
        if success:
            # Set labels
            if row == 2:  # Bottom row
                ax.set_xlabel("Time Step", fontsize=13, fontweight='bold')
            if col == 0:  # Left column
                ax.set_ylabel("Yaw angle (deg)", fontsize=13, fontweight='bold')

    # Add column titles
    col_titles = ["Dataset 1", "Dataset 2", "Dataset 3"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    # # Add row labels on left side
    # row_labels = ["Baseline", "+ Council", "AIVV"]
    # for row, row_label in enumerate(row_labels):
    #     axes[row, 0].set_ylabel(row_label, fontsize=13, fontweight='bold', labelpad=15)
    
    # Adjust layout
    legend_handles = [
        Line2D([0], [0], color="gray", linewidth=1.0, label="Data from IMU (Yaw)"),
        Line2D([0], [0], color="red", linewidth=1.0, label="Predicted Yaw"),
        Patch(facecolor="#8ecae6", alpha=0.35, label="Calibrated Prediction Interval"),
        Line2D([0], [0], color="royalblue", linestyle="--", linewidth=1.0, label="Fault Detected"),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save figure
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Combined FDI figure saved to: {output_file}")
        plt.close()
        return True
    except Exception as e:
        print(f"Error saving figure: {e}")
        plt.close()
        return False


# ============================================================================
# ARGUMENT PARSING & MAIN
# ============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate plots from ACA run results (single or combined 3x3, version-aware)"
    )
    
    # Mode 1: Single plot
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='Path to single run directory (generates individual plots)'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default='general',
        choices=['general', 'math_only', 'no_inspector_tuner'],
        help='Plot version for single-run mode (default: general)'
    )
    
    # Mode 2: Combined 3x3 figure
    parser.add_argument(
        '--run-dirs',
        type=str,
        nargs=9,
        default=None,
        help='Paths to 9 run directories (generates combined 3x3 figure)'
    )
    
    parser.add_argument(
        '--versions',
        type=str,
        nargs=9,
        default=None,
        help='Versions for 9 runs in --run-dirs mode (default: all "general")'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='combined_plot.png',
        help='Output filename for combined figure (default: combined_plot.png)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='aca_',
        help='Plot filename prefix for single plot mode (default: aca_)'
    )
    
    parser.add_argument(
        '--fault-persistence',
        type=int,
        default=1,
        help='Minimum consecutive FAIL steps (default: 1)'
    )
    
    parser.add_argument(
        '--default-jobs',
        action='store_true',
        help='Use default 9 jobs for combined figure'
    )
    
    args = parser.parse_args()
    
    # Determine which mode to use
    if args.run_dir:
        # Single plot mode
        print("=" * 70)
        print(f"Regenerating plots for: {args.run_dir}")
        print(f"Version: {args.version}")
        print("=" * 70)
        
        success = plot_single_detailed(
            args.run_dir,
            version=args.version,
            prefix=args.prefix,
            fault_persistence=args.fault_persistence
        )
        
        if not success:
            sys.exit(1)
    
    elif args.run_dirs:
        # Combined 3x3 mode
        run_dirs = [Path(d) for d in args.run_dirs]
        
        # Default versions
        if args.versions is None:
            versions = ['general'] * 9
        else:
            versions = args.versions
        
        # Create job dictionaries
        jobs = []
        for idx, (run_dir, version) in enumerate(zip(run_dirs, versions)):
            job = {
                "label": f"({chr(97 + idx)})",
                "version": version,
                "run_dir": str(run_dir),
                "prefix": "aca_",
            }
            jobs.append(job)
        
        success = create_combined_figure(
            jobs,
            output_file=args.output_file,
            fault_persistence=args.fault_persistence
        )
        
        if not success:
            sys.exit(1)
    
    elif args.default_jobs:
        # Use default 9 jobs
        print("=" * 70)
        print("Generating combined figure with default 9 jobs")
        print("=" * 70)
        
        success = create_combined_figure(
            DEFAULT_JOBS,
            output_file=args.output_file,
            fault_persistence=args.fault_persistence
        )
        
        if not success:
            sys.exit(1)
    
    else:
        # No arguments provided
        parser.print_help()
        print("\n" + "=" * 70)
        print("USAGE EXAMPLES:")
        print("=" * 70)
        print("\n1. Generate plots for a single run (general version):")
        print('   python regenerate_plots_batch.py --run-dir "/path/to/run"')
        print("\n2. Generate plots for a single run (math_only version):")
        print('   python regenerate_plots_batch.py --run-dir "/path/to/run" --version math_only')
        print("\n3. Generate plots for a single run (no_inspector_tuner version):")
        print('   python regenerate_plots_batch.py --run-dir "/path/to/run" --version no_inspector_tuner')
        print("\n4. Generate combined 3x3 figure from 9 runs:")
        print('   python regenerate_plots_batch.py --run-dirs path1 path2 ... path9')
        print("\n5. Generate combined 3x3 figure with specified versions:")
        print('   python regenerate_plots_batch.py --run-dirs path1 path2 ... path9 --versions v1 v2 ... v9')
        print("\n6. Generate with default 9 jobs:")
        print('   python regenerate_plots_batch.py --default-jobs')
        print("=" * 70)


if __name__ == "__main__":
    main()
