#!/usr/bin/env python
"""
AIVV Ablation Study Script

Runs ablation experiments for the COLM 2026 paper:
  Mode 1: math_sentry_only     — Math engine + Sentry gate only (no council/tuner)
  Mode 2: no_inspector_tuner   — Math + Sentry + Council only (no Inspector/Tuner)

Usage:
    python ablation.py --mode math_sentry_only --epochs 10 --seed 42 --plot
    python ablation.py --mode no_inspector_tuner --epochs 10 --seed 42 --plot
    python ablation.py --mode all --epochs 10 --seed 42 --plot
"""

import argparse
import atexit
import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress verbose httpx/httpcore logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))

from aivv.config import ACAConfig
from aivv.engine.engine import UUVEngine
from aivv.orchestrator import ACAOrchestrator
from aivv.data.uuv_loader import UUVDataLoader
from aivv.agents.base import Decision, CaseFile, AgentResult, AgentType
from aivv.agents.sentry import SentryAgent
from aivv.agents.requirements_engineer import RequirementsEngineerAgent
from aivv.agents.failure_manager import FailureManagerAgent
from aivv.agents.system_engineer import SystemEngineerAgent
from aivv.llm.genai_client import get_llm_client
from aivv.evaluation.metrics import ACAMetrics
from torch.utils.data import DataLoader, TensorDataset

# Reuse helpers from main.py
from main import (
    set_seed,
    setup_logging,
    train_engine,
    resolve_uuv_fault_window,
    save_inference_plots,
    save_uuv_fdi_style_plot,
    apply_fault_persistence,
    cumulative_prf,
    run_inference,
    _TeeStream,
)


# ---------------------------------------------------------------------------
# Terminal logging (same as main.py)
# ---------------------------------------------------------------------------

def setup_run_terminal_logging(
    base_dir: Optional[Path] = None,
    run_dir: Optional[Path] = None,
) -> Path:
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

    def _cleanup():
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        finally:
            if not log_file.closed:
                log_file.close()

    atexit.register(_cleanup)
    return run_dir


# ============================================================================
# Mode 1: Math + Sentry Only — no council, no tuner
# ============================================================================

def run_math_sentry_only(
    config: ACAConfig,
    engine: UUVEngine,
    data_loader: UUVDataLoader,
    max_samples: Optional[int],
    label_overrides: Optional[np.ndarray],
    fault_persistence: int,
    run_log_dir: Path,
) -> Dict[str, Any]:
    """Mode 1: Sentry conformal check is the final decision. No LLM calls."""

    # Clone engine so we don't pollute state
    mode_engine = engine.clone()
    mode_config = copy.deepcopy(config)
    mode_config.ENABLE_AGENT_3 = False
    mode_config.ENABLE_GROUP_B = False

    orchestrator = ACAOrchestrator(
        config=mode_config,
        engine=mode_engine,
        run_log_dir=str(run_log_dir),
    )

    test_loader = data_loader.get_test_loader()

    trace = {
        "index": [], "prediction": [], "actual": [], "error": [], "bound": [],
        "uncertainty": [], "uncertainty_bound": [],
        "initial_fail": [], "raw_council_fail": [], "final_fail": [],
        "pred_label": [], "true_label": [],
    }

    t0 = time.perf_counter()
    sample_count = 0

    # Fine-tune history (not used here, but process_sample expects it)
    history_window = None
    history_targets = None
    try:
        train_X, train_y = data_loader.get_train_data()
        history_window = train_X.to(mode_engine.device)
        history_targets = train_y.to(mode_engine.device)
    except Exception:
        pass

    print("\n[MODE 1] Math + Sentry Only — no LLM council")
    print("=" * 60)

    for batch_x, batch_y in test_loader:
        for i in range(len(batch_x)):
            if max_samples and sample_count >= max_samples:
                break

            curr_window = batch_x[i:i + 1].to(mode_engine.device)
            actual_val = float(batch_y[i].item())

            result = orchestrator.process_sample(
                window=curr_window,
                actual=actual_val,
                sample_id=sample_count,
                history_window=history_window,
                history_targets=history_targets,
                actual_label_override=(
                    None if label_overrides is None or sample_count >= len(label_overrides)
                    else int(label_overrides[sample_count])
                ),
            )

            is_fail = result.final_decision == Decision.FAIL
            trace["index"].append(sample_count)
            trace["prediction"].append(float(result.case_file.prediction))
            trace["actual"].append(float(result.case_file.actual))
            trace["error"].append(float(result.case_file.error))
            trace["bound"].append(float(result.case_file.bound))
            trace["uncertainty"].append(float(result.case_file.uncertainty))
            ub = float("nan")
            if result.case_file.agent_results:
                sp = result.case_file.agent_results[0].payload or {}
                if "uncertainty_bound" in sp:
                    ub = float(sp["uncertainty_bound"])
            trace["uncertainty_bound"].append(ub)
            trace["initial_fail"].append(result.initial_decision == Decision.FAIL)
            trace["raw_council_fail"].append(result.raw_final_decision == Decision.FAIL)
            trace["final_fail"].append(is_fail)
            trace["pred_label"].append(1 if is_fail else 0)

            if label_overrides is not None and sample_count < len(label_overrides):
                trace["true_label"].append(int(label_overrides[sample_count]))
            else:
                trace["true_label"].append(float("nan"))

            if (sample_count + 1) % 50 == 0:
                m = orchestrator.get_metrics()
                print(
                    f"  Processed {sample_count + 1} samples — "
                    f"F1: {m.f1_score:.4f}, Recall: {m.recall:.4f}"
                )

            sample_count += 1

        if max_samples and sample_count >= max_samples:
            break

    elapsed = time.perf_counter() - t0
    print(f"\n[MODE 1] Done: {sample_count} samples in {elapsed:.1f}s, 0 LLM calls")

    return {
        "mode": "math_sentry_only",
        "mode_label": "Math + Sentry Only (no council/tuner)",
        "trace": trace,
        "metrics": orchestrator.get_metrics(),
        "elapsed_s": elapsed,
        "llm_calls": 0,
        "total_tokens": 0,
        "sample_count": sample_count,
    }


# ============================================================================
# Mode 2: No Inspector / No Tuner — Council-only deliberation
# ============================================================================

def run_no_inspector_tuner(
    config: ACAConfig,
    engine: UUVEngine,
    data_loader: UUVDataLoader,
    max_samples: Optional[int],
    label_overrides: Optional[np.ndarray],
    fault_persistence: int,
    run_log_dir: Path,
) -> Dict[str, Any]:
    """Mode 2: Sentry + Council only. No Inspector or Tuner.

    Pipeline per sample:
      1. Sentry → PASS → done (PASS).
      2. Sentry → FAIL → Council (A4/A5/A6) votes → 2-of-3 majority.
         - Majority FAIL → confirmed anomaly (FAIL).
         - Majority PASS → override Sentry (final PASS).
    No adaptation loop, no clone-and-verify. Single council round.
    """

    mode_engine = engine.clone()
    mode_config = copy.deepcopy(config)

    # Initialize Sentry (rule-based, no LLM)
    sentry = SentryAgent(mode_engine, mode_config)

    # Initialize council LLM agents
    llm_req = get_llm_client(
        base_url=mode_config.llm_base_url,
        model=mode_config.agent4_model,
        api_key=mode_config.llm_api_key,
        temperature=mode_config.llm_temperature,
        max_tokens=mode_config.llm_max_tokens,
    )
    llm_fail = get_llm_client(
        base_url=mode_config.llm_base_url,
        model=mode_config.agent5_model,
        api_key=mode_config.llm_api_key,
        temperature=mode_config.llm_temperature,
        max_tokens=mode_config.llm_max_tokens,
    )
    llm_sys = get_llm_client(
        base_url=mode_config.llm_base_url,
        model=mode_config.agent6_model,
        api_key=mode_config.llm_api_key,
        temperature=mode_config.llm_temperature,
        max_tokens=mode_config.llm_max_tokens,
    )

    req_engineer = RequirementsEngineerAgent(llm_req, mode_config)
    failure_manager = FailureManagerAgent(llm_fail, mode_config)
    system_engineer = SystemEngineerAgent(llm_sys, mode_config)

    # Metrics
    metrics = ACAMetrics()

    test_loader = data_loader.get_test_loader()

    trace = {
        "index": [], "prediction": [], "actual": [], "error": [], "bound": [],
        "uncertainty": [], "uncertainty_bound": [],
        "initial_fail": [], "raw_council_fail": [], "final_fail": [],
        "pred_label": [], "true_label": [],
    }

    t0 = time.perf_counter()
    sample_count = 0
    llm_calls = 0
    council_activations = 0
    council_overrides = 0
    council_confirmations = 0

    print("\n[MODE 2] No Inspector / No Tuner — Council-only deliberation")
    print("=" * 60)

    # Get training evidence for council context
    training_evidence = mode_engine.get_training_evidence()

    for batch_x, batch_y in test_loader:
        for i in range(len(batch_x)):
            if max_samples and sample_count >= max_samples:
                break

            curr_window = batch_x[i:i + 1].to(mode_engine.device)
            actual_val = float(batch_y[i].item())

            # --- Step 1: Sentry check ---
            case_file = CaseFile(sample_id=sample_count)
            sentry_result = sentry.execute(case_file, window=curr_window, actual=actual_val)
            initial_decision = sentry_result.decision

            if sentry_result.decision == Decision.PASS:
                # Sentry PASS → final PASS (no council needed)
                final_decision = Decision.PASS
                was_overridden = False
            else:
                # --- Step 2: Council vote ---
                council_activations += 1

                # Build context payload for council agents
                deviation_ratio = case_file.error / case_file.bound if case_file.bound > 0 else float('inf')
                sample_json = {
                    'sample_id': case_file.sample_id,
                    'prediction': case_file.prediction,
                    'actual': case_file.actual,
                    'error': case_file.error,
                    'bound': case_file.bound,
                    'alpha': case_file.alpha,
                    'uncertainty': case_file.uncertainty,
                    'deviation_ratio': deviation_ratio,
                    'current_error': case_file.error,
                    'system_mode': 'FAULT_SUSPECT',
                }

                # Add training evidence (compact)
                if training_evidence:
                    train_probe = training_evidence.get('model_baseline', {}).get('train_probe', {})
                    raw_window_baseline = training_evidence.get('raw_window_baseline', {})
                    sample_json['training_evidence'] = {
                        'model_baseline': {
                            'train_probe': {
                                'error_distribution': {
                                    'q95': train_probe.get('error_distribution', {}).get('q95'),
                                    'q99': train_probe.get('error_distribution', {}).get('q99'),
                                },
                                'uncertainty_distribution': {
                                    'q95': train_probe.get('uncertainty_distribution', {}).get('q95'),
                                    'q99': train_probe.get('uncertainty_distribution', {}).get('q99'),
                                },
                            }
                        },
                        'raw_window_baseline': {
                            'abs_value_distribution': {
                                'q95': raw_window_baseline.get('abs_value_distribution', {}).get('q95'),
                                'q99': raw_window_baseline.get('abs_value_distribution', {}).get('q99'),
                            },
                            'window_range_distribution': {
                                'q95': raw_window_baseline.get('window_range_distribution', {}).get('q95'),
                                'q99': raw_window_baseline.get('window_range_distribution', {}).get('q99'),
                            },
                            'window_max_abs_step_distribution': {
                                'q95': raw_window_baseline.get('window_max_abs_step_distribution', {}).get('q95'),
                                'q99': raw_window_baseline.get('window_max_abs_step_distribution', {}).get('q99'),
                            },
                        },
                    }

                # Add frame-level window context
                try:
                    # Build raw window for rich context
                    raw_window = None
                    if getattr(data_loader, "use_differencing", False):
                        raw_np = data_loader.get_test_raw_window(sample_count)
                        if raw_np is not None:
                            raw_window = raw_np.reshape(-1).tolist()
                    else:
                        win_np = curr_window.detach().cpu().numpy().reshape(-1, 1)
                        raw_np = data_loader.scaler.inverse_transform(win_np).reshape(-1)
                        raw_window = raw_np.tolist()

                    if raw_window is not None:
                        w = torch.tensor(raw_window, dtype=torch.float32)
                        frame_values = [float(v) for v in raw_window]
                        if w.numel() >= 2:
                            d = w[1:] - w[:-1]
                            max_abs_step = float(torch.max(torch.abs(d)).item())
                            mean_abs_step = float(torch.mean(torch.abs(d)).item())
                        else:
                            max_abs_step = 0.0
                            mean_abs_step = 0.0

                        sample_json['frame_values'] = frame_values
                        sample_json['frame_stats'] = {
                            'window_size': len(frame_values),
                            'min': float(min(frame_values)),
                            'max': float(max(frame_values)),
                            'mean': float(np.mean(frame_values)),
                            'std': float(np.std(frame_values)),
                            'range': float(max(frame_values) - min(frame_values)),
                            'max_abs_value': float(max(abs(v) for v in frame_values)),
                            'first_value': frame_values[0],
                            'last_value': frame_values[-1],
                            'delta_last_first': frame_values[-1] - frame_values[0],
                            'max_abs_step_delta': max_abs_step,
                            'mean_abs_step_delta': mean_abs_step,
                        }
                except Exception:
                    pass

                # Send compact payload (no frame_values) to A4/A6,
                # full payload to A5 (Failure Manager)
                sample_json_compact = dict(sample_json)
                sample_json_compact.pop('frame_values', None)

                print(f"  [COUNCIL] Sample {sample_count}: Running council agents...", flush=True)

                _t = time.perf_counter()
                req_result = req_engineer.execute(case_file, sample_json_compact)
                llm_calls += 1
                print(f"    Req. Engineer done in {time.perf_counter() - _t:.3f}s", flush=True)

                _t = time.perf_counter()
                failure_result = failure_manager.execute(case_file, sample_json)
                llm_calls += 1
                print(f"    Failure Manager done in {time.perf_counter() - _t:.3f}s", flush=True)

                _t = time.perf_counter()
                syseng_result = system_engineer.execute(case_file, sample_json_compact)
                llm_calls += 1
                print(f"    System Engineer done in {time.perf_counter() - _t:.3f}s", flush=True)

                council_votes = [req_result, failure_result, syseng_result]

                # --- Step 3: 2-of-3 majority ---
                fail_votes = sum(1 for v in council_votes if v.decision == Decision.FAIL)
                pass_votes = len(council_votes) - fail_votes

                if fail_votes >= 2:
                    final_decision = Decision.FAIL
                    was_overridden = False  # Council confirms Sentry's FAIL
                    council_confirmations += 1
                    print(
                        f"    Council majority FAIL ({fail_votes}/3) → confirmed anomaly",
                        flush=True,
                    )
                else:
                    final_decision = Decision.PASS
                    was_overridden = True  # Council overrides Sentry's FAIL
                    council_overrides += 1
                    print(
                        f"    Council majority PASS ({pass_votes}/3) → override Sentry → PASS",
                        flush=True,
                    )

            # --- Record results ---
            is_fail = final_decision == Decision.FAIL
            prediction = 1 if is_fail else 0

            if label_overrides is not None and sample_count < len(label_overrides):
                true_label = int(label_overrides[sample_count])
                council_decision = (
                    "OVERRIDE" if was_overridden
                    else ("CONFIRM" if prediction == 1 else None)
                ) if initial_decision == Decision.FAIL else None
                metrics.update(prediction, true_label, council_decision)
            else:
                true_label = float("nan")

            trace["index"].append(sample_count)
            trace["prediction"].append(float(case_file.prediction))
            trace["actual"].append(float(case_file.actual))
            trace["error"].append(float(case_file.error))
            trace["bound"].append(float(case_file.bound))
            trace["uncertainty"].append(float(case_file.uncertainty))
            ub = float("nan")
            if case_file.agent_results:
                sp = case_file.agent_results[0].payload or {}
                if "uncertainty_bound" in sp:
                    ub = float(sp["uncertainty_bound"])
            trace["uncertainty_bound"].append(ub)
            trace["initial_fail"].append(initial_decision == Decision.FAIL)
            trace["raw_council_fail"].append(is_fail)
            trace["final_fail"].append(is_fail)
            trace["pred_label"].append(prediction)
            trace["true_label"].append(true_label)

            if (sample_count + 1) % 50 == 0:
                print(
                    f"  Processed {sample_count + 1} samples — "
                    f"F1: {metrics.f1_score:.4f}, Recall: {metrics.recall:.4f}"
                )

            sample_count += 1

        if max_samples and sample_count >= max_samples:
            break

    elapsed = time.perf_counter() - t0
    print(
        f"\n[MODE 2] Done: {sample_count} samples in {elapsed:.1f}s, "
        f"{llm_calls} LLM calls ({council_activations} council activations)"
    )
    print(
        f"  Council overrides: {council_overrides}, "
        f"confirmations: {council_confirmations}"
    )

    return {
        "mode": "no_inspector_tuner",
        "mode_label": "Council Only (no Inspector/Tuner)",
        "trace": trace,
        "metrics": metrics,
        "elapsed_s": elapsed,
        "llm_calls": llm_calls,
        "total_tokens": 0,
        "sample_count": sample_count,
        "council_activations": council_activations,
        "council_overrides": council_overrides,
        "council_confirmations": council_confirmations,
    }


# ============================================================================
# Mode 3: Full AIVV — all agents enabled
# ============================================================================

def run_full_aivv(
    config: ACAConfig,
    engine: UUVEngine,
    data_loader: UUVDataLoader,
    max_samples: Optional[int],
    label_overrides: Optional[np.ndarray],
    fault_persistence: int,
    run_log_dir: Path,
) -> Dict[str, Any]:
    """Mode 3: Full AIVV pipeline with all agents enabled."""

    mode_engine = engine.clone()
    mode_config = copy.deepcopy(config)
    mode_config.ENABLE_AGENT_3 = True
    mode_config.ENABLE_GROUP_B = True

    orchestrator = ACAOrchestrator(
        config=mode_config,
        engine=mode_engine,
        run_log_dir=str(run_log_dir),
    )

    test_loader = data_loader.get_test_loader()

    print("\n[MODE 3] Full AIVV — all agents enabled")
    print("=" * 60)

    t0 = time.perf_counter()
    trace = run_inference(
        orchestrator=orchestrator,
        test_loader=test_loader,
        max_samples=max_samples,
        data_loader=data_loader,
        fault_persistence=fault_persistence,
        label_overrides=label_overrides,
    )
    elapsed = time.perf_counter() - t0

    sample_count = len(trace["index"])

    # Count LLM calls from agent comm logger
    llm_calls = 0
    try:
        llm_calls = orchestrator.comm_logger.get_total_communications()
    except Exception:
        pass

    print(f"\n[MODE 3] Done: {sample_count} samples in {elapsed:.1f}s")

    return {
        "mode": "full_aivv",
        "mode_label": "Full AIVV",
        "trace": trace,
        "metrics": orchestrator.get_metrics(),
        "elapsed_s": elapsed,
        "llm_calls": llm_calls,
        "total_tokens": 0,
        "sample_count": sample_count,
    }


# ============================================================================
# Output: ablation.txt and plots
# ============================================================================

def write_ablation_summary(
    results: List[Dict[str, Any]],
    out_path: Path,
    args,
    config: ACAConfig,
) -> None:
    """Write ablation.txt with per-mode stats and a comparison table."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("AIVV Ablation Study Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"uuv_file: {args.uuv_file}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"samples: {args.samples or 'all'}\n")
        f.write(f"modes_run: {[r['mode'] for r in results]}\n")
        f.write("\n")

        f.write("Shared Hyperparameters\n")
        f.write("-" * 70 + "\n")
        f.write(f"window_size: {config.window_size}\n")
        f.write(f"lstm_hidden_dim: {config.lstm_hidden_dim}\n")
        f.write(f"lstm_num_layers: {config.lstm_num_layers}\n")
        f.write(f"lstm_dropout: {config.lstm_dropout}\n")
        f.write(f"mc_samples: {config.mc_samples}\n")
        f.write(f"default_alpha: {config.default_alpha}\n")
        f.write(f"fine_tune_lr: {config.fine_tune_lr}\n")
        f.write("\n")

        # Per-mode details
        for r in results:
            m = r["metrics"]
            f.write("=" * 70 + "\n")
            f.write(f"MODE: {r['mode_label']}\n")
            f.write(f"  Key: {r['mode']}\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Samples processed: {r['sample_count']}\n")
            f.write(f"  Wall-clock time:   {r['elapsed_s']:.2f} s\n")
            f.write(f"  LLM API calls:     {r['llm_calls']}\n")
            if r.get("council_activations") is not None:
                f.write(f"  Council activations: {r['council_activations']}\n")
            if r.get("council_overrides") is not None:
                f.write(f"  Council overrides:   {r['council_overrides']}\n")
            if r.get("council_confirmations") is not None:
                f.write(f"  Council confirmations: {r['council_confirmations']}\n")
            f.write("\n")
            f.write("  Confusion Matrix\n")
            f.write(f"                  Predicted PASS   Predicted FAIL\n")
            f.write(f"    Actual PASS      {m.true_negatives:<12d}   {m.false_positives}\n")
            f.write(f"    Actual FAIL      {m.false_negatives:<12d}   {m.true_positives}\n")
            f.write("\n")
            f.write(f"  Precision:            {m.precision:.6f}\n")
            f.write(f"  Recall:               {m.recall:.6f}\n")
            f.write(f"  F1-Score:             {m.f1_score:.6f}\n")
            f.write(f"  False Positive Rate:  {m.false_positive_rate:.6f}\n")
            f.write(f"  Accuracy:             {m.accuracy:.6f}\n")
            f.write(f"  Specificity:          {m.specificity:.6f}\n")
            if hasattr(m, "overrides"):
                f.write(f"  Council overrides:    {m.overrides}\n")
                f.write(f"  Council confirmations:{m.confirmations}\n")
                f.write(f"  Override rate:        {m.override_rate:.6f}\n")
            f.write("\n")

        # Comparison table
        if len(results) > 1:
            f.write("=" * 70 + "\n")
            f.write("COMPARISON TABLE\n")
            f.write("=" * 70 + "\n")
            header = (
                f"{'Mode':<30s} {'F1':>8s} {'Recall':>8s} {'Prec':>8s} "
                f"{'FPR':>8s} {'Time(s)':>9s} {'LLM#':>7s}"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for r in results:
                m = r["metrics"]
                row = (
                    f"{r['mode']:<30s} {m.f1_score:>8.4f} {m.recall:>8.4f} "
                    f"{m.precision:>8.4f} {m.false_positive_rate:>8.4f} "
                    f"{r['elapsed_s']:>9.2f} {r['llm_calls']:>7d}"
                )
                f.write(row + "\n")
            f.write("\n")

    print(f"Ablation summary written to: {out_path}")


def save_mode_plots(
    result: Dict[str, Any],
    data_loader: UUVDataLoader,
    out_dir: Path,
    config: ACAConfig,
    fault_persistence: int,
) -> None:
    """Save plots for a single ablation mode."""
    trace = result["trace"]
    mode = result["mode"]
    prefix = f"{mode}_"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Standard inference plots with mode prefix
    save_inference_plots(trace, out_dir=out_dir, fault_persistence=fault_persistence)

    # Save trace to JSON for later regeneration
    trace_json = {
        "index": [int(x) for x in trace["index"]],
        "prediction": [float(x) for x in trace["prediction"]],
        "actual": [float(x) for x in trace["actual"]],
        "error": [float(x) for x in trace["error"]],
        "bound": [float(x) for x in trace["bound"]],
        "uncertainty": [float(x) for x in trace["uncertainty"]],
        "uncertainty_bound": [float(x) for x in trace["uncertainty_bound"]],
        "initial_fail": [bool(x) for x in trace["initial_fail"]],
        "raw_council_fail": [bool(x) for x in trace["raw_council_fail"]],
        "final_fail": [bool(x) for x in trace["final_fail"]],
        "pred_label": [int(x) for x in trace["pred_label"]],
        "true_label": [float(x) if not np.isnan(x) else None for x in trace["true_label"]],
    }
    trace_file = out_dir / "trace.json"
    with open(trace_file, 'w') as f:
        json.dump(trace_json, f, indent=2)

    # Rename the plots to include mode prefix
    standard_plots = [
        "aca_pred_vs_actual.png",
        "aca_residual_vs_bound.png",
        "aca_uncertainty_vs_bound.png",
        "aca_decision_timeline.png",
        "aca_cumulative_prf1.png",
    ]
    for plot_name in standard_plots:
        src = out_dir / plot_name
        dst = out_dir / f"{prefix}{plot_name}"
        if src.exists():
            src.rename(dst)

    # UUV FDI-style plot
    try:
        save_uuv_fdi_style_plot(
            trace,
            data_loader=data_loader,
            out_dir=out_dir,
            alpha=config.default_alpha,
            fault_persistence=fault_persistence,
            output_filename=f"{prefix}uuv_fdi_style.png",
            prediction_label=f"Predicted Yaw ({mode})",
            title=f"FDI of Yaw Rate ({result['mode_label']})",
            fail_marker_mode="all",
        )
    except Exception as e:
        print(f"  Warning: Could not save UUV FDI plot for {mode}: {e}")


def save_comparison_plot(
    results: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Save a bar chart comparing F1, Recall, Precision, FPR across modes."""
    if len(results) < 2:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    mode_names = [r["mode"] for r in results]
    f1_scores = [r["metrics"].f1_score for r in results]
    recalls = [r["metrics"].recall for r in results]
    precisions = [r["metrics"].precision for r in results]
    fprs = [r["metrics"].false_positive_rate for r in results]

    x = np.arange(len(mode_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 1.5 * width, f1_scores, width, label="F1-Score", color="#2196F3")
    ax.bar(x - 0.5 * width, recalls, width, label="Recall", color="#4CAF50")
    ax.bar(x + 0.5 * width, precisions, width, label="Precision", color="#FF9800")
    ax.bar(x + 1.5 * width, fprs, width, label="FPR", color="#F44336")

    ax.set_ylabel("Score")
    ax.set_title("AIVV Ablation Study: Detection Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_comparison_metrics.png", dpi=160)
    plt.close()

    # Time comparison
    times = [r["elapsed_s"] for r in results]
    llm_calls = [r["llm_calls"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.bar(mode_names, times, color="#673AB7", alpha=0.85)
    ax1.set_ylabel("Wall-clock Time (s)")
    ax1.set_title("Computational Cost: Time")
    ax1.tick_params(axis="x", rotation=15)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(mode_names, llm_calls, color="#009688", alpha=0.85)
    ax2.set_ylabel("LLM API Calls")
    ax2.set_title("Computational Cost: LLM Usage")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "ablation_comparison_cost.png", dpi=160)
    plt.close()

    print(f"Comparison plots saved to: {out_dir}")


def save_combined_fdi_plot(
    results: List[Dict[str, Any]],
    data_loader: UUVDataLoader,
    out_dir: Path,
    alpha: float = 0.05,
    fault_persistence: int = 1,
) -> None:
    """Save a single combined FDI plot overlaying all ablation modes.

    Anomaly markers use mode-specific colors:
        Mode 1 (math_sentry_only): yellow
        Mode 2 (no_inspector_tuner): green
        Mode 3 (full_aivv): blue
    """
    if not results:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Color and linestyle mapping per mode (distinct dashes to avoid overlap masking)
    #   "--"  = dashed       ------
    #   "-."  = dashdot      -·-·-·
    #   ":"   = dotted       ······
    MODE_STYLES = {
        "math_sentry_only": ("#FFD700", "--",  "Math+Sentry Only"),
        "no_inspector_tuner": ("#4CAF50", "-.", "Council Only (no Insp./Tuner)"),
        "full_aivv": ("#9C27B0", ":",  "Full AIVV"),
    }

    # Full UUV series (shared background)
    t_all = np.asarray(data_loader.time_all, dtype=np.float32)
    y_all = np.asarray(data_loader.yaw_all, dtype=np.float32)

    scale = float(getattr(data_loader.scaler, "scale_", np.array([1.0], dtype=np.float32))[0])

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(t_all, y_all, color="gray", linewidth=1.0, label="Data from IMU (Yaw)")

    # Use the last result (highest fidelity) for prediction line + band
    ref = results[-1]
    ref_trace = ref["trace"]
    pred_s = np.asarray(ref_trace.get("prediction", []), dtype=np.float32)
    bound_s = np.asarray(ref_trace.get("bound", []), dtype=np.float32)

    if len(pred_s) > 0:
        pred_native = data_loader.scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
        band = bound_s * scale

        if getattr(data_loader, "use_differencing", False):
            prev = data_loader.test_prev_raw.detach().cpu().numpy().reshape(-1)
            m_ref = min(len(prev), len(pred_native), len(band))
            pred_native = pred_native[:m_ref]
            band = band[:m_ref]
            pred_plot = prev[:m_ref] + pred_native
        else:
            pred_plot = pred_native

        upper = pred_plot + band
        lower = pred_plot - band

        start = int(data_loader.n_train_raw + data_loader.target_index_offset)
        if getattr(data_loader, "use_differencing", False):
            start += 1
        stop = min(start + len(pred_plot), len(t_all))
        m_ref = max(0, stop - start)

        if m_ref > 0:
            t_pred = t_all[start:stop]
            pred_plot = pred_plot[:m_ref]
            upper = upper[:m_ref]
            lower = lower[:m_ref]

            ax.plot(t_pred, pred_plot, color="red", linewidth=1.0, alpha=0.9, label="Predicted Yaw")
            ax.fill_between(
                t_pred, lower, upper,
                color="#8ecae6", alpha=0.35,
                label=f"Calibrated interval ({(1.0 - alpha) * 100:.0f}%)",
            )

    # Overlay anomaly markers for each mode
    for r in results:
        mode = r["mode"]
        color, lstyle, mode_label = MODE_STYLES.get(mode, ("#9E9E9E", "--", mode))
        trace = r["trace"]
        fail_flags = np.asarray(trace.get("final_fail", []), dtype=bool)

        if len(fail_flags) == 0:
            continue

        persistent_fail = apply_fault_persistence(fail_flags, fault_persistence)

        # Map to global timeline
        start = int(data_loader.n_train_raw + data_loader.target_index_offset)
        if getattr(data_loader, "use_differencing", False):
            start += 1
        stop = min(start + len(persistent_fail), len(t_all))
        m = max(0, stop - start)
        if m == 0:
            continue

        t_mode = t_all[start:stop]
        pf = persistent_fail[:m]

        labeled = False
        for tp in t_mode[pf]:
            ax.axvline(
                tp, color=color, linestyle=lstyle, linewidth=1.0, alpha=0.85,
                label=mode_label if not labeled else None,
            )
            labeled = True

    ax.set_title("FDI of Yaw Rate (Ablation Comparison)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Yaw Rate")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_combined_fdi.png", dpi=160)
    plt.close()
    print(f"Combined FDI plot saved to: {out_dir / 'ablation_combined_fdi.png'}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="AIVV Ablation Study")

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["math_sentry_only", "no_inspector_tuner", "full_aivv", "all"],
        help="Ablation mode to run",
    )
    parser.add_argument("--dataset", type=str, default="uuv", choices=["uuv"])
    parser.add_argument("--uuv-file", type=str,
                        default=str(Path(__file__).resolve().parent / "data" / "uuv" / "UUV_yaw_fault_2.txt"))
    parser.add_argument("--uuv-downsample", type=int, default=20)
    parser.add_argument("--uuv-train-frac", type=float, default=0.7)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--uuv-use-differencing", action="store_true")
    parser.add_argument("--uuv-fault-start-index", type=int, default=None)
    parser.add_argument("--uuv-fault-end-index", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--lstm-num-layers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--fault-persistence", type=int, default=3)
    parser.add_argument("--run-log-dir", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--disable-council-temporal-filter", action="store_true",
        help="Disable council temporal debouncing (uses raw council decision each step)",
    )
    parser.add_argument(
        "--sentry-uncertainty-only-fail", action="store_true",
        help="If set, Sentry can trigger FAIL on high uncertainty alone",
    )
    parser.add_argument(
        "--council-confirm-k", type=int, default=2,
        help="Council temporal filter: consecutive FAILs required to enter failure mode",
    )
    parser.add_argument(
        "--council-release-k", type=int, default=2,
        help="Council temporal filter: consecutive PASSes required to exit failure mode",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which modes to run
    if args.mode == "all":
        modes = ["math_sentry_only", "no_inspector_tuner", "full_aivv"]
    else:
        modes = [args.mode]

    # Setup logging
    run_log_dir = setup_run_terminal_logging(
        run_dir=Path(args.run_log_dir) if args.run_log_dir else None,
    )
    print(f"Ablation run logs: {run_log_dir.resolve()}")

    setup_logging(args.verbose)

    # Config
    config = ACAConfig()
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

    config.domain = "uuv"
    config.lstm_input_dim = 1
    config.council_confirm_k = int(args.council_confirm_k)
    config.council_release_k = int(args.council_release_k)
    config.ENABLE_COUNCIL_TEMPORAL_FILTER = not bool(args.disable_council_temporal_filter)
    config.sentry_uncertainty_only_fail = bool(args.sentry_uncertainty_only_fail)

    device = config.get_device()
    print(f"Using device: {device}")

    print("=" * 70)
    print("AIVV ABLATION STUDY")
    print(f"Modes: {modes}")
    print("=" * 70)

    # ---- Load data (shared across all modes) ----
    print("\nLoading data...")
    data_loader = UUVDataLoader(
        file_path=args.uuv_file,
        window_size=config.window_size,
        train_ratio=args.uuv_train_frac,
        downsample=args.uuv_downsample,
        horizon=args.forecast_horizon,
        use_differencing=args.uuv_use_differencing,
        device=config.get_device(),
    )

    label_overrides = None
    fault_window = resolve_uuv_fault_window(
        file_path=args.uuv_file,
        start_index=args.uuv_fault_start_index,
        end_index=args.uuv_fault_end_index,
    )
    if fault_window is not None:
        label_overrides = data_loader.build_test_fault_labels(*fault_window)
        pos = int(np.sum(label_overrides))
        print(
            f"Fault labels: global indices {fault_window[0]}-{fault_window[1]} "
            f"→ {pos} labeled test windows"
        )
    else:
        print("No explicit UUV fault labels configured.")

    print(f"Dataset stats: {data_loader.get_stats()}")

    # ---- Train shared engine ----
    print("\nTraining shared LSTM engine...")
    engine = UUVEngine(config=config)

    train_X, train_y = data_loader.get_train_data()
    n_cal = max(1, int(0.2 * len(train_X)))
    cal_X, cal_y = train_X[-n_cal:], train_y[-n_cal:]
    fit_X, fit_y = train_X[:-n_cal], train_y[:-n_cal]

    train_loader = DataLoader(
        TensorDataset(fit_X, fit_y), batch_size=32, shuffle=True
    )
    train_engine(engine, train_loader, epochs=args.epochs, cal_data=(cal_X, cal_y))
    engine.register_training_reference(
        train_windows=fit_X,
        train_targets=fit_y,
        calibration_windows=cal_X,
        calibration_targets=cal_y,
        raw_train_series=getattr(data_loader, "x_train_raw", None),
        dataset_stats=data_loader.get_stats(),
        scaler_stats={
            "mean": [float(v) for v in np.asarray(
                getattr(data_loader.scaler, "mean_", np.array([], dtype=np.float32))
            ).reshape(-1)],
            "scale": [float(v) for v in np.asarray(
                getattr(data_loader.scaler, "scale_", np.array([], dtype=np.float32))
            ).reshape(-1)],
        },
    )

    print("\nShared engine trained and calibrated.")
    print(f"Conformal bound (α={config.default_alpha}): "
          f"{engine.get_conformal_interval(config.default_alpha):.6f}")

    # ---- Run each mode ----
    all_results: List[Dict[str, Any]] = []

    for mode in modes:
        # Re-seed for reproducibility across modes
        set_seed(config.SEED)

        mode_run_dir = run_log_dir / mode
        mode_run_dir.mkdir(parents=True, exist_ok=True)

        if mode == "math_sentry_only":
            result = run_math_sentry_only(
                config=config,
                engine=engine,
                data_loader=data_loader,
                max_samples=args.samples,
                label_overrides=label_overrides,
                fault_persistence=args.fault_persistence,
                run_log_dir=mode_run_dir,
            )
        elif mode == "no_inspector_tuner":
            result = run_no_inspector_tuner(
                config=config,
                engine=engine,
                data_loader=data_loader,
                max_samples=args.samples,
                label_overrides=label_overrides,
                fault_persistence=args.fault_persistence,
                run_log_dir=mode_run_dir,
            )
        elif mode == "full_aivv":
            result = run_full_aivv(
                config=config,
                engine=engine,
                data_loader=data_loader,
                max_samples=args.samples,
                label_overrides=label_overrides,
                fault_persistence=args.fault_persistence,
                run_log_dir=mode_run_dir,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        all_results.append(result)

        # Save mode-specific plots
        if args.plot:
            plot_dir = mode_run_dir / "plots"
            save_mode_plots(
                result, data_loader, plot_dir, config, args.fault_persistence
            )

        # Print mode metrics
        m = result["metrics"]
        print(f"\n{'─' * 50}")
        print(f"  {result['mode_label']}")
        print(f"  F1={m.f1_score:.4f}  Recall={m.recall:.4f}  "
              f"Precision={m.precision:.4f}  FPR={m.false_positive_rate:.4f}")
        print(f"  Time={result['elapsed_s']:.1f}s  LLM calls={result['llm_calls']}")
        print(f"{'─' * 50}\n")

    # ---- Write ablation.txt ----
    ablation_txt_path = run_log_dir / "ablation.txt"
    write_ablation_summary(all_results, ablation_txt_path, args, config)

    # ---- Comparison plots (when multiple modes) ----
    if args.plot and len(all_results) > 1:
        save_comparison_plot(all_results, run_log_dir / "comparison_plots")
        save_combined_fdi_plot(
            all_results, data_loader, run_log_dir / "comparison_plots",
            alpha=config.default_alpha,
            fault_persistence=args.fault_persistence,
        )

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print(f"Results: {ablation_txt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
