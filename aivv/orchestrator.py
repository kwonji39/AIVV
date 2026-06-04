"""
ACA Orchestrator

Main workflow coordinator that manages the interaction between all agents.

Pipeline Flow:
  Sentry(FAIL) → [A4, A5, A6 parallel] → Inspector(vote collector, majority 2/3)
  → if FAIL: Inspector→Tuner → Sentry(re-eval)
  → if still FAIL: [A4, A5, A6] → Inspector (2nd loop, max 2)
  → final decision
"""

import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
import time

from .config import ACAConfig
from .engine.engine import UUVEngine
from .agents.base import CaseFile, Decision, AgentResult, AgentType
from .agents.sentry import SentryAgent
from .agents.inspector import InspectorAgent
from .agents.tuner import TunerAgent
from .agents.requirements_engineer import RequirementsEngineerAgent
from .agents.failure_manager import FailureManagerAgent
from .agents.system_engineer import SystemEngineerAgent
from .llm.genai_client import get_llm_client
from .evaluation.metrics import ACAMetrics
from .utils.json_logger import ACALogger


@dataclass
class ProcessingResult:
    """Result of processing a single sample."""
    sample_id: int
    initial_decision: Decision
    raw_final_decision: Decision
    final_decision: Decision
    is_anomaly: bool
    was_overridden: bool
    case_file: CaseFile
    engine_updated: bool


class ACAOrchestrator:
    """
    Adaptive Council Architecture Orchestrator

    Pipeline:
      1. Sentry checks sample → PASS exits immediately.
      2. On FAIL: send sample JSON to A4, A5, A6 in parallel.
      3. Inspector collects 3 votes → 2/3 majority.
         - If PASS → accept (override Sentry).
         - If FAIL → Inspector determines Tuner params.
      4. Tuner executes adaptation → sends result to Sentry only.
      5. Sentry re-evaluates.
         - If PASS → done.
         - If still FAIL → 2nd council loop (steps 2-3, max 2 loops).
      6. After 2 failed loops → final FAIL (confirmed anomaly).

    Ablation modes:
    - ENABLE_AGENT_3: Toggle Tuner (Experiment A)
    - ENABLE_GROUP_B: Toggle Council (Experiment B)
    """

    MAX_COUNCIL_LOOPS = 2

    def __init__(
        self,
        config: Optional[ACAConfig] = None,
        engine: Optional[UUVEngine] = None,
        run_log_dir: Optional[str] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: ACA configuration
            engine: Pre-initialized engine (optional)
            run_log_dir: Optional explicit directory for per-run logs/artifacts
        """
        self.config = config or ACAConfig()

        # Initialize engine
        self.engine = engine or UUVEngine(config=self.config)

        # Initialize LLM clients for each agent (multi-LLM architecture)
        llm_inspector = get_llm_client(
            base_url=self.config.llm_base_url,
            model=self.config.agent2_model,
            api_key=self.config.llm_api_key,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        llm_req_engineer = get_llm_client(
            base_url=self.config.llm_base_url,
            model=self.config.agent4_model,
            api_key=self.config.llm_api_key,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        llm_failure_mgr = get_llm_client(
            base_url=self.config.llm_base_url,
            model=self.config.agent5_model,
            api_key=self.config.llm_api_key,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        llm_sys_engineer = get_llm_client(
            base_url=self.config.llm_base_url,
            model=self.config.agent6_model,
            api_key=self.config.llm_api_key,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        llm_tuner = get_llm_client(
            base_url=self.config.llm_base_url,
            model=self.config.agent3_model,
            api_key=self.config.llm_api_key,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )

        # Initialize agents
        self.sentry = SentryAgent(self.engine, self.config)
        self.inspector = InspectorAgent(llm_inspector, self.config)
        self.tuner = TunerAgent(self.engine, self.config, llm_client=llm_tuner)
        self.req_engineer = RequirementsEngineerAgent(llm_req_engineer, self.config)
        self.failure_manager = FailureManagerAgent(llm_failure_mgr, self.config)
        self.system_engineer = SystemEngineerAgent(llm_sys_engineer, self.config)

        # Keep reference to default client for orchestrator use
        self.llm_client = llm_inspector

        # Metrics tracking
        self.metrics = ACAMetrics()

        # Agent Communication Logger
        from .utils.agent_comm_logger import AgentCommLogger
        self.comm_logger = AgentCommLogger(base_dir="logs", run_dir=run_log_dir)
        if self.engine.get_training_evidence():
            self.comm_logger.save_run_artifact(
                "training_evidence",
                self.engine.get_training_evidence(),
            )

        # JSON Logger for agent decision tracking
        self.json_logger = ACALogger(
            config=self.config,
            logs_dir=self.comm_logger.get_run_dir(),
            job_id=f"aca_run_{self.comm_logger.timestamp}"
        )

        # Model mapping for logging
        self.agent_models = {
            AgentType.SENTRY: "rule-based",
            AgentType.INSPECTOR: self.config.agent2_model,
            AgentType.TUNER: self.config.agent3_model,
            AgentType.REQUIREMENTS_ENGINEER: self.config.agent4_model,
            AgentType.FAILURE_MANAGER: self.config.agent5_model,
            AgentType.SYSTEM_ENGINEER: self.config.agent6_model,
        }

        # Logging
        self.logger = logging.getLogger("ACAOrchestrator")
        self.processing_history: List[ProcessingResult] = []

        # Online temporal filter state for Council outcomes
        self._council_fail_streak = 0
        self._council_pass_streak = 0
        self._council_failure_latched = False
    
    def process_sample(
        self,
        window: torch.Tensor,
        actual: float,
        sample_id: int,
        history_window: Optional[torch.Tensor] = None,
        history_targets: Optional[torch.Tensor] = None,
        actual_label_override: Optional[int] = None,
        raw_window: Optional[torch.Tensor] = None,
    ) -> ProcessingResult:
        """
        Process a single sample through the ACA pipeline.

        Flow:
          1. Sentry → PASS exits.
          2. On FAIL: A4/A5/A6 vote → Inspector collects → majority 2/3.
          3. If majority PASS → override Sentry, accept.
          4. If majority FAIL → Inspector→Tuner → Sentry re-eval.
          5. If Sentry re-eval PASS → done.
          6. If still FAIL → 2nd council loop (max 2 total).
          7. After 2 failed loops → confirmed anomaly.
        """
        case_file = CaseFile(sample_id=sample_id)

        # ========== STEP 1: Sentry Check ==========
        sentry_result = self.sentry.execute(case_file, window=window, actual=actual)
        initial_decision = sentry_result.decision

        if sentry_result.decision == Decision.PASS:
            self._finalize_result(
                case_file, final_decision=Decision.PASS, actual=actual,
                was_overridden=False, engine_updated=False,
                actual_label_override=actual_label_override,
            )
            return ProcessingResult(
                sample_id=sample_id, initial_decision=initial_decision,
                raw_final_decision=Decision.PASS, final_decision=Decision.PASS,
                is_anomaly=False, was_overridden=False,
                case_file=case_file, engine_updated=False
            )

        # ========== SENTRY FAIL: Begin adaptation pipeline ==========

        # Ablation: Agent 3 disabled → immediate FAIL
        if not self.config.ENABLE_AGENT_3:
            self._finalize_result(
                case_file, final_decision=Decision.FAIL, actual=actual,
                was_overridden=False, engine_updated=False,
                actual_label_override=actual_label_override,
            )
            return ProcessingResult(
                sample_id=sample_id, initial_decision=initial_decision,
                raw_final_decision=Decision.FAIL, final_decision=Decision.FAIL,
                is_anomaly=True, was_overridden=False,
                case_file=case_file, engine_updated=False
            )

        # Set current sample in comm logger
        self.comm_logger.set_sample(sample_id)

        # Ablation: Council disabled → go straight to Inspector/Tuner fallback
        if not self.config.ENABLE_GROUP_B:
            return self._process_without_council(
                case_file, window, actual, sample_id,
                initial_decision, history_window, history_targets,
                actual_label_override,
            )

        # ========== COUNCIL LOOP (max 2 iterations) ==========
        for loop_idx in range(self.MAX_COUNCIL_LOOPS):
            print(f"  [PIPELINE] Council loop {loop_idx + 1}/{self.MAX_COUNCIL_LOOPS}", flush=True)

            # ---- STEP 2: Send sample to A4, A5, A6 in parallel ----
            sample_json = self._build_sample_json(
                case_file,
                window=raw_window if raw_window is not None else window,
            )
            self.comm_logger.save_sample_artifact(
                f"council_context_loop_{loop_idx + 1}",
                {
                    'loop': loop_idx + 1,
                    'payload': sample_json,
                }
            )
            # Keep A5 (Failure Manager) rich context for frame-level checks,
            # but send a compact payload to A4/A6 to reduce JSON-generation failures.
            sample_json_compact = dict(sample_json)
            sample_json_compact.pop('frame_values', None)

            print("  [COUNCIL] Running Requirements Engineer...", flush=True)
            _a4_t0 = time.perf_counter()
            req_result = self.req_engineer.execute(case_file, sample_json_compact)
            print(f"  [COUNCIL] Requirements Engineer done in {time.perf_counter() - _a4_t0:.3f} s", flush=True)

            print("  [COUNCIL] Running Failure Manager...", flush=True)
            _a5_t0 = time.perf_counter()
            failure_result = self.failure_manager.execute(case_file, sample_json)
            print(f"  [COUNCIL] Failure Manager done in {time.perf_counter() - _a5_t0:.3f} s", flush=True)

            print("  [COUNCIL] Running System Engineer...", flush=True)
            _a6_t0 = time.perf_counter()
            syseng_result = self.system_engineer.execute(case_file, sample_json_compact)
            print(f"  [COUNCIL] System Engineer done in {time.perf_counter() - _a6_t0:.3f} s", flush=True)

            council_votes = [req_result, failure_result, syseng_result]

            # Log council → inspector communication
            self.comm_logger.log_communication(
                from_agent="council",
                to_agent="inspector",
                payload={
                    'loop': loop_idx + 1,
                    'votes': [
                        {'agent': v.agent_type.value, 'vote': v.decision.value,
                         'confidence': v.confidence, 'reasoning': v.reasoning}
                        for v in council_votes
                    ]
                }
            )

            # ---- STEP 3: Orchestrator computes majority (2/3) ----
            fail_votes = sum(1 for v in council_votes if v.decision == Decision.FAIL)
            pass_votes = len(council_votes) - fail_votes
            raw_council_decision = Decision.FAIL if fail_votes >= 2 else Decision.PASS

            # Update temporal reporting state (informational only).
            # Control flow uses the raw council majority, not the debounced state.
            reporting_decision = self._apply_council_temporal_filter(raw_council_decision)
            if reporting_decision != raw_council_decision:
                print(
                    f"  [COUNCIL TEMPORAL REPORT] "
                    f"raw={raw_council_decision.value} -> reporting={reporting_decision.value} "
                    f"(fail_streak={self._council_fail_streak}, "
                    f"pass_streak={self._council_pass_streak}, "
                    f"latched={self._council_failure_latched})"
                )

            if raw_council_decision == Decision.FAIL:
                # Raw council majority FAIL overrides Sentry → confirmed anomaly (no Inspector call needed)
                self.tuner.discard_temp_engine()
                self._log_final_council(
                    Decision.FAIL, council_votes,
                    inspector_result=None,
                    loop_idx=loop_idx, promote=False,
                    raw_council_decision=raw_council_decision,
                    reporting_decision=reporting_decision,
                )
                self._finalize_result(
                    case_file, final_decision=Decision.FAIL, actual=actual,
                    was_overridden=True, engine_updated=False,
                    actual_label_override=actual_label_override,
                )
                return ProcessingResult(
                    sample_id=sample_id, initial_decision=initial_decision,
                    raw_final_decision=raw_council_decision,
                    final_decision=Decision.FAIL,
                    is_anomaly=True, was_overridden=True,
                    case_file=case_file, engine_updated=False
                )

            # ---- STEP 4 (final loop only): Council PASS on last loop → final PASS ----
            is_final_loop = (loop_idx == self.MAX_COUNCIL_LOOPS - 1)
            if is_final_loop and raw_council_decision == Decision.PASS:
                # Second council majority PASS after failed adaptation → accept as PASS
                print(
                    f"  [PIPELINE] Loop {loop_idx + 1} (final) council majority PASS "
                    f"→ final PASS (council override, skipping Inspector/Tuner)",
                    flush=True,
                )
                self.tuner.discard_temp_engine()
                self._log_final_council(
                    Decision.PASS, council_votes,
                    inspector_result=None,
                    loop_idx=loop_idx, promote=False,
                    raw_council_decision=raw_council_decision,
                    reporting_decision=reporting_decision,
                )
                self._finalize_result(
                    case_file, final_decision=Decision.PASS, actual=actual,
                    was_overridden=True, engine_updated=False,
                    actual_label_override=actual_label_override,
                )
                return ProcessingResult(
                    sample_id=sample_id, initial_decision=initial_decision,
                    raw_final_decision=raw_council_decision,
                    final_decision=Decision.PASS,
                    is_anomaly=False, was_overridden=True,
                    case_file=case_file, engine_updated=False
                )

            # ---- STEP 4: Majority PASS → Inspector determines tuner params ----
            inspector_result = self.inspector.execute(
                case_file,
                council_votes=council_votes,
                pass_votes=pass_votes,
                fail_votes=fail_votes,
            )

            instruction = inspector_result.payload
            self.comm_logger.log_communication(
                from_agent="inspector", to_agent="tuner",
                payload=instruction
            )

            tuner_result = self.tuner.execute(
                case_file,
                instruction=instruction,
                window=window,
                history_window=history_window,
                history_targets=history_targets,
                comm_logger=self.comm_logger,
            )

            # ---- STEP 5: Tuner → Sentry re-evaluation ----
            reevaluation_candidate = self.tuner.get_reevaluation_candidate()
            adaptation_payload = self.tuner.get_adaptation_payload()

            if reevaluation_candidate:
                reeval_result = self.sentry.reevaluate(
                    case_file,
                    new_prediction=reevaluation_candidate.new_prediction,
                    new_bound=reevaluation_candidate.new_bound,
                    new_uncertainty=reevaluation_candidate.new_uncertainty,
                    new_alpha=reevaluation_candidate.applied_alpha,
                )
                case_file.add_result(reeval_result)
                print(
                    f"  [SENTRY RE-EVAL] loop={loop_idx + 1} "
                    f"new_error={reevaluation_candidate.new_error:.4f} bound={reevaluation_candidate.new_bound:.4f} "
                    f"→ {reeval_result.decision.value}",
                    flush=True,
                )

                if reeval_result.decision == Decision.PASS:
                    # Tuner fixed it → promote engine, done
                    self.tuner.promote_temp_engine()
                    self._apply_recommended_alpha(adaptation_payload)
                    self.comm_logger.save_sample_artifact(
                        f"promoted_training_evidence_loop_{loop_idx + 1}",
                        self.engine.get_training_evidence(),
                    )
                    self._log_final_council(
                        Decision.PASS, council_votes,
                        inspector_result=inspector_result,
                        loop_idx=loop_idx, promote=True,
                        raw_council_decision=raw_council_decision,
                        reporting_decision=reporting_decision,
                    )
                    self._finalize_result(
                        case_file, final_decision=Decision.PASS, actual=actual,
                        was_overridden=True, engine_updated=True,
                        actual_label_override=actual_label_override,
                    )
                    return ProcessingResult(
                        sample_id=sample_id, initial_decision=initial_decision,
                        raw_final_decision=raw_council_decision,
                        final_decision=Decision.PASS,
                        is_anomaly=False, was_overridden=True,
                        case_file=case_file, engine_updated=True
                    )

            # Sentry re-eval still FAIL — discard temp engine, loop again
            self.tuner.discard_temp_engine()

        # ========== Exhausted all loops → confirmed anomaly ==========
        print(
            f"  [PIPELINE] Max {self.MAX_COUNCIL_LOOPS} council loops exhausted → FAIL",
            flush=True,
        )
        self._log_final_council(
            Decision.FAIL, council_votes,
            inspector_result=inspector_result,
            loop_idx=self.MAX_COUNCIL_LOOPS - 1, promote=False,
            raw_council_decision=raw_council_decision,
            reporting_decision=reporting_decision,
        )
        final_decision = Decision.FAIL
        self._finalize_result(
            case_file, final_decision=final_decision, actual=actual,
            was_overridden=False, engine_updated=False,
            actual_label_override=actual_label_override,
        )
        return ProcessingResult(
            sample_id=sample_id, initial_decision=initial_decision,
            raw_final_decision=Decision.FAIL,
            final_decision=final_decision,
            is_anomaly=True, was_overridden=False,
            case_file=case_file, engine_updated=False
        )

    # ------------------------------------------------------------------
    # Helper: process without council (Experiment B ablation)
    # ------------------------------------------------------------------
    def _process_without_council(
        self,
        case_file: CaseFile,
        window: torch.Tensor,
        actual: float,
        sample_id: int,
        initial_decision: Decision,
        history_window: Optional[torch.Tensor],
        history_targets: Optional[torch.Tensor],
        actual_label_override: Optional[int],
    ) -> ProcessingResult:
        """Fallback path when council is disabled (Experiment B)."""
        # Default instruction: TRY_BOTH
        instruction = {
            'action': 'TRY_BOTH',
            'new_alpha': self.config.recalibration_alpha,
            'epochs': self.config.fine_tune_epochs,
            'learning_rate': self.config.fine_tune_lr,
            'reasoning': 'Council disabled — automatic TRY_BOTH.',
        }
        tuner_result = self.tuner.execute(
            case_file,
            instruction=instruction,
            window=window,
            history_window=history_window,
            history_targets=history_targets,
            comm_logger=self.comm_logger,
        )
        reevaluation_candidate = self.tuner.get_reevaluation_candidate()
        adaptation_payload = self.tuner.get_adaptation_payload()

        if reevaluation_candidate and reevaluation_candidate.passes_reevaluation:
            self.tuner.promote_temp_engine()
            self._apply_recommended_alpha(adaptation_payload)
            final_decision = Decision.PASS
            was_overridden = True
            engine_updated = True
        else:
            self.tuner.discard_temp_engine()
            final_decision = Decision.FAIL
            was_overridden = False
            engine_updated = False

        self._finalize_result(
            case_file, final_decision=final_decision, actual=actual,
            was_overridden=was_overridden, engine_updated=engine_updated,
            actual_label_override=actual_label_override,
        )
        return ProcessingResult(
            sample_id=sample_id, initial_decision=initial_decision,
            raw_final_decision=final_decision, final_decision=final_decision,
            is_anomaly=(final_decision == Decision.FAIL),
            was_overridden=was_overridden,
            case_file=case_file, engine_updated=engine_updated
        )

    # ------------------------------------------------------------------
    # Helper: build sample JSON for council agents
    # ------------------------------------------------------------------
    def _build_sample_json(
        self,
        case_file: CaseFile,
        window: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Build the JSON payload that Sentry sends to each council agent."""
        deviation_ratio = case_file.error / case_file.bound if case_file.bound > 0 else float('inf')
        payload = {
            'sample_id': case_file.sample_id,
            'prediction': case_file.prediction,
            'actual': case_file.actual,
            'error': case_file.error,
            'bound': case_file.bound,
            'alpha': case_file.alpha,
            'uncertainty': case_file.uncertainty,
            'deviation_ratio': deviation_ratio,
            'current_error': case_file.error,
            # V&V context: agents are only called after Sentry FAIL
            'system_mode': 'FAULT_SUSPECT',
        }

        base_training_evidence = self.engine.get_training_evidence()
        if base_training_evidence:
            payload['training_evidence'] = self._compact_training_evidence(base_training_evidence)

        latest_tuner = self._get_latest_tuner_adaptation_payload(case_file)
        if latest_tuner:
            for key in (
                'action_taken', 'old_bound', 'new_bound', 'old_error', 'new_error',
                'recommended_alpha', 'applied_alpha', 'improvement'
            ):
                if key in latest_tuner:
                    payload[key] = latest_tuner[key]
            if 'candidate_training_evidence' in latest_tuner:
                payload['candidate_training_evidence'] = self._compact_training_evidence(
                    latest_tuner['candidate_training_evidence']
                )

        latest_reeval = self._get_latest_reevaluation_payload(case_file)
        if latest_reeval:
            payload['reevaluation'] = latest_reeval
            payload['reevaluation_error'] = latest_reeval.get('new_error')
            payload['reevaluation_bound'] = latest_reeval.get('new_bound')
            payload['reevaluation_prediction'] = latest_reeval.get('new_prediction')
            payload['reevaluation_pass'] = bool(
                latest_reeval.get('new_error', float('inf')) <= latest_reeval.get('new_bound', 0.0)
            )

        # Add frame-level sequence context (20-step window) for gradient/trend checks.
        if window is not None:
            try:
                w = window.detach().float().reshape(-1).cpu()
                if w.numel() > 0:
                    frame_values = [float(v.item()) for v in w]
                    payload['frame_values'] = frame_values

                    if w.numel() >= 2:
                        d = w[1:] - w[:-1]
                        max_abs_step = float(torch.max(torch.abs(d)).item())
                        mean_abs_step = float(torch.mean(torch.abs(d)).item())
                    else:
                        max_abs_step = 0.0
                        mean_abs_step = 0.0

                    payload['frame_stats'] = {
                        'window_size': int(w.numel()),
                        'min': float(torch.min(w).item()),
                        'max': float(torch.max(w).item()),
                        'mean': float(torch.mean(w).item()),
                        'std': float(torch.std(w, unbiased=False).item()) if w.numel() > 1 else 0.0,
                        'range': float((torch.max(w) - torch.min(w)).item()),
                        'max_abs_value': float(torch.max(torch.abs(w)).item()),
                        'first_value': float(w[0].item()),
                        'last_value': float(w[-1].item()),
                        'delta_last_first': float((w[-1] - w[0]).item()),
                        'max_abs_step_delta': max_abs_step,
                        'mean_abs_step_delta': mean_abs_step,
                    }
            except Exception:
                # Keep council pipeline robust even if extra diagnostics fail.
                pass

        return payload

    def _compact_training_evidence(self, evidence: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Keep only the baseline statistics used by council agents."""
        if not isinstance(evidence, dict):
            return {}

        train_probe = evidence.get('model_baseline', {}).get('train_probe', {})
        raw_window_baseline = evidence.get('raw_window_baseline', {})

        compact = {
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
        return compact

    def _get_latest_tuner_adaptation_payload(self, case_file: CaseFile) -> Optional[Dict[str, Any]]:
        """Return the latest tuner adaptation payload if available."""
        for result in reversed(case_file.agent_results):
            if result.agent_type == AgentType.TUNER:
                payload = result.payload or {}
                adaptation_payload = payload.get('adaptation_payload')
                if isinstance(adaptation_payload, dict):
                    return adaptation_payload
        return None

    def _get_latest_reevaluation_payload(self, case_file: CaseFile) -> Optional[Dict[str, Any]]:
        """Return latest Sentry reevaluation payload if available."""
        for result in reversed(case_file.agent_results):
            if result.agent_type == AgentType.SENTRY:
                payload = result.payload or {}
                if payload.get('reevaluation'):
                    return payload
        return None

    # ------------------------------------------------------------------
    # Helper: log final council decision to comm logger
    # ------------------------------------------------------------------
    def _log_final_council(
        self,
        final_decision: Decision,
        council_votes: List[AgentResult],
        inspector_result: Optional[AgentResult],
        loop_idx: int,
        promote: bool,
        raw_council_decision: Optional[Decision] = None,
        reporting_decision: Optional[Decision] = None,
    ) -> None:
        """Log final council decision to JSON comm logger."""
        pass_votes = sum(1 for v in council_votes if v.decision == Decision.PASS)
        self.comm_logger.log_final_decision({
            'decision': final_decision.value,
            'promote_temp_engine': promote,
            'council_loop': loop_idx + 1,
            'raw_council_decision': (
                raw_council_decision.value if raw_council_decision is not None else None
            ),
            'temporal_reporting_decision': (
                reporting_decision.value if reporting_decision is not None else None
            ),
            'temporal_filter_applied': (
                raw_council_decision is not None
                and reporting_decision is not None
                and raw_council_decision != reporting_decision
            ),
            'inspector_reasoning': (
                inspector_result.reasoning if inspector_result is not None
                else f"Raw council majority PASS ({pass_votes}/{len(council_votes)}). Inspector skipped."
            ),
            'vote_breakdown': {
                'requirements_engineer': council_votes[0].decision.value if len(council_votes) > 0 else None,
                'failure_manager': council_votes[1].decision.value if len(council_votes) > 1 else None,
                'system_engineer': council_votes[2].decision.value if len(council_votes) > 2 else None,
                'pass_votes': pass_votes,
                'total_votes': len(council_votes),
                'rule': 'majority_2_of_3',
            }
        })

    def _apply_recommended_alpha(self, adaptation_payload: Optional[Any]) -> None:
        """Apply Agent-3 recommended alpha to operating engine/Sentry when approved."""
        if not bool(getattr(self.config, "adaptive_alpha_enabled", True)):
            return
        if adaptation_payload is None:
            return

        rec = getattr(adaptation_payload, "recommended_alpha", None)
        if rec is None:
            rec = getattr(adaptation_payload, "applied_alpha", None)
        if rec is None:
            return

        amin = float(getattr(self.config, "adaptive_alpha_min", 0.01))
        amax = float(getattr(self.config, "adaptive_alpha_max", 0.10))
        smoothing = float(getattr(self.config, "adaptive_alpha_smoothing", 1.0))
        smoothing = max(0.0, min(1.0, smoothing))

        current_alpha = float(self.engine.conformal.alpha)
        target_alpha = max(amin, min(amax, float(rec)))
        new_alpha = (1.0 - smoothing) * current_alpha + smoothing * target_alpha
        new_alpha = max(amin, min(amax, new_alpha))

        applied = self.engine.set_operating_alpha(new_alpha)
        self.sentry.alpha = float(applied)
        print(
            f"  [ADAPTIVE ALPHA] current={current_alpha:.3f} target={target_alpha:.3f} "
            f"applied={applied:.3f} (smoothing={smoothing:.2f})",
            flush=True,
        )

    def _apply_council_temporal_filter(self, decision: Decision) -> Decision:
        """
        Update debounced Council reporting state online.

        This is reporting-only state and does not alter pipeline control flow.

        - Enter reporting failure mode only after `council_confirm_k` consecutive FAILs.
        - Exit reporting failure mode only after `council_release_k` consecutive PASSes.
        """
        if not self.config.ENABLE_GROUP_B:
            return decision
        if not getattr(self.config, "ENABLE_COUNCIL_TEMPORAL_FILTER", True):
            return decision

        confirm_k = max(1, int(getattr(self.config, "council_confirm_k", 3)))
        release_k = max(1, int(getattr(self.config, "council_release_k", 2)))

        is_fail = (decision == Decision.FAIL)
        if is_fail:
            self._council_fail_streak += 1
            self._council_pass_streak = 0
        else:
            self._council_pass_streak += 1
            self._council_fail_streak = 0

        if not self._council_failure_latched:
            if is_fail and self._council_fail_streak >= confirm_k:
                self._council_failure_latched = True
                return Decision.FAIL
            return Decision.PASS

        # already latched in failure mode
        if (not is_fail) and self._council_pass_streak >= release_k:
            self._council_failure_latched = False
            return Decision.PASS
        return Decision.FAIL
    
    def _finalize_result(
        self,
        case_file: CaseFile,
        final_decision: Decision,
        actual: float,
        was_overridden: bool,
        engine_updated: bool,
        actual_label_override: Optional[int] = None,
    ) -> None:
        """Finalize processing and update metrics."""
        # Convert decision to prediction label
        prediction = 1 if final_decision == Decision.FAIL else 0
        
        actual_label = None if actual_label_override is None else int(actual_label_override)
        
        # Update metrics
        council_decision = "OVERRIDE" if was_overridden else "CONFIRM" if prediction == 1 else None
        if actual_label is not None:
            self.metrics.update(prediction, actual_label, council_decision)
        
        # Log sample start
        self.json_logger.log_sample_start(case_file.sample_id, actual_label)
        
        # Log all agent decisions for this sample
        for agent_result in case_file.agent_results:
            model_used = self.agent_models.get(agent_result.agent_type, "unknown")
            self.json_logger.log_agent_decision(
            sample_id=case_file.sample_id,
                agent_name=agent_result.agent_type.value,
                agent_type=str(agent_result.agent_type),
                model_used=model_used,
                decision=agent_result.decision.value,
                confidence=agent_result.confidence,
                reasoning=agent_result.reasoning,
                payload=agent_result.payload
            )
        
        # Log sample completion
        correct = None if actual_label is None else (prediction == actual_label)
        self.json_logger.log_sample_complete(
            sample_id=case_file.sample_id,
            final_decision=final_decision.value,
            was_override=was_overridden,
            correct=correct
        )
        
        # Store in history
        self.processing_history.append(ProcessingResult(
            sample_id=case_file.sample_id,
            initial_decision=Decision.FAIL if case_file.error > case_file.bound else Decision.PASS,
            raw_final_decision=final_decision,
            final_decision=final_decision,
            is_anomaly=(final_decision == Decision.FAIL),
            was_overridden=was_overridden,
            case_file=case_file,
            engine_updated=engine_updated
        ))
    
    def process_batch(
        self,
        windows: torch.Tensor,
        actuals: torch.Tensor,
        history_windows: Optional[torch.Tensor] = None,
        history_targets: Optional[torch.Tensor] = None
    ) -> List[ProcessingResult]:
        """
        Process a batch of samples.
        
        Args:
            windows: Batch of windows (batch, seq_len, features)
            actuals: Batch of actual values (batch,)
            history_windows: Optional historical data
            history_targets: Optional historical targets
            
        Returns:
            List of ProcessingResult for each sample
        """
        results = []
        
        for i in range(len(windows)):
            window = windows[i:i+1]  # Keep batch dimension
            actual = float(actuals[i].item()) if hasattr(actuals[i], 'item') else float(actuals[i])
            
            result = self.process_sample(
                window=window,
                actual=actual,
                sample_id=i,
                history_window=history_windows,
                history_targets=history_targets
            )
            results.append(result)
        
        return results
    
    def get_metrics(self) -> ACAMetrics:
        """Get current metrics."""
        return self.metrics
    
    def print_metrics_report(self) -> str:
        """Generate metrics report and log to JSON."""
        report = self.metrics.print_report()
        
        # Log summary to JSON
        summary = self.metrics.get_summary()
        council_stats = {
            'overrides': summary.get('council_overrides', 0),
            'confirmations': summary.get('council_confirmations', 0),
            'override_rate': summary.get('override_rate', 0.0)
        }
        self.json_logger.log_summary(summary, council_stats)
        
        return report
    
    def reset_metrics(self) -> None:
        """Reset metrics for new experiment."""
        self.metrics.reset()
        self.processing_history = []

    def get_vv_summary(self) -> Dict[str, Any]:
        """
        Aggregate V&V verdicts from all council invocations across the run.

        Returns a dict with human-readable verdicts for:
          - requirements_engineer: Normal Mode V&V summary
          - failure_manager: Failure Mode V&V summary
        """
        re_total = re_fail = 0
        fm_total = fm_fail = 0
        se_total = se_fail = 0
        re_fail_samples: List[Dict[str, Any]] = []
        fm_fail_samples: List[Dict[str, Any]] = []
        se_tuning_proposals: List[Dict[str, Any]] = []

        for proc in self.processing_history:
            cf = proc.case_file
            for r in cf.agent_results:
                payload = r.payload or {}
                if r.agent_type == AgentType.REQUIREMENTS_ENGINEER:
                    re_total += 1
                    if r.decision.value == 'FAIL':
                        re_fail += 1
                        re_fail_samples.append({
                            'sample_id': cf.sample_id,
                            'veto_reason': payload.get('veto_reason') or '',
                            'requirement_section': payload.get('requirement_section') or '',
                            'reasoning': r.reasoning or '',
                        })
                elif r.agent_type == AgentType.FAILURE_MANAGER:
                    fm_total += 1
                    if r.decision.value == 'FAIL':
                        fm_fail += 1
                        fm_metrics = payload.get('failure_metrics', {})
                        fm_fail_samples.append({
                            'sample_id': cf.sample_id,
                            'peak_deviation': fm_metrics.get('peak_deviation'),
                            'is_converging': fm_metrics.get('is_converging'),
                            'oscillation_count': fm_metrics.get('oscillation_count'),
                            'settling_trend': fm_metrics.get('settling_trend'),
                            'fm_assessment': payload.get('failure_management_assessment') or '',
                            'reasoning': r.reasoning or '',
                        })
                elif r.agent_type == AgentType.SYSTEM_ENGINEER:
                    se_total += 1
                    if r.decision.value == 'FAIL':
                        se_fail += 1
                    # Only collect proposals when FM or RE voted FAIL on this sample
                    fm_voted_fail = any(
                        prev.agent_type == AgentType.FAILURE_MANAGER
                        and prev.decision.value == 'FAIL'
                        for prev in cf.agent_results
                    )
                    re_voted_fail = any(
                        prev.agent_type == AgentType.REQUIREMENTS_ENGINEER
                        and prev.decision.value == 'FAIL'
                        for prev in cf.agent_results
                    )
                    if fm_voted_fail or re_voted_fail:
                        tuning_proposal = payload.get('proposed_gains')
                        if tuning_proposal:
                            se_tuning_proposals.append({
                                'sample_id': cf.sample_id,
                                'vote': r.decision.value,
                                'tuning_proposal': tuning_proposal,
                                'tuning_reasoning': payload.get('tuning_reasoning', ''),
                                'reasoning': r.reasoning or '',
                                'triggered_by': (
                                    'FM+RE' if (fm_voted_fail and re_voted_fail)
                                    else 'FM' if fm_voted_fail else 'RE'
                                ),
                            })

        # --- Requirements Engineer verdict ---
        if re_total == 0:
            re_verdict = "No council invocations recorded (council may be disabled)."
        elif re_fail == 0:
            re_verdict = (
                f"PASS  -- All {re_total} sampled windows verified against operational requirements. "
                "Yaw magnitude, window range, step size, and bound multiplier remained within policy limits throughout the run."
            )
        else:
            # Pick the most informative failure sample (first occurrence)
            top = re_fail_samples[0]
            veto = top.get('veto_reason') or top.get('reasoning') or 'requirement violated'
            # Trim to a readable length
            veto = veto[:220].rstrip()
            re_verdict = (
                f"FAIL  -- {re_fail}/{re_total} sampled windows violated operational requirements. "
                f"First violation at sample {top['sample_id']}: {veto}"
            )

        # --- Failure Manager verdict ---
        if fm_total == 0:
            fm_verdict = "No council invocations recorded (council may be disabled)."
        elif fm_fail == 0:
            fm_verdict = (
                f"PASS  -- All {fm_total} fault-suspect windows satisfied failure management requirements. "
                "Failure effects were contained, system responses were converging, and oscillation counts remained within limits."
            )
        else:
            top = fm_fail_samples[0]
            peak = top.get('peak_deviation')
            converging = top.get('is_converging')
            osc = top.get('oscillation_count')
            assess = top.get('fm_assessment') or top.get('reasoning') or ''
            assess = assess[:220].rstrip()

            details = []
            if peak is not None:
                details.append(f"peak_deviation={peak:.2f}")
            if converging is not None:
                details.append("response=CONVERGING" if converging else "response=DIVERGING")
            if osc is not None:
                details.append(f"oscillation_count={osc}")
            detail_str = (", ".join(details) + ". " if details else "")

            fm_verdict = (
                f"FAIL  -- {fm_fail}/{fm_total} fault-suspect windows violated failure management requirements. "
                f"First violation at sample {top['sample_id']}: {detail_str}{assess}"
            )

        return {
            'requirements_engineer': {
                'total_calls': re_total,
                'fail_count': re_fail,
                'test_result': re_verdict,
                'fail_samples': re_fail_samples[:5],  # cap at 5 for summary
            },
            'failure_manager': {
                'total_calls': fm_total,
                'fail_count': fm_fail,
                'test_result': fm_verdict,
                'fail_samples': fm_fail_samples[:5],
            },
            'system_engineer': {
                'total_calls': se_total,
                'fail_count': se_fail,
                'tuning_proposals': se_tuning_proposals,
            },
        }

    
    def set_ablation_mode(
        self,
        enable_agent_3: bool = True,
        enable_group_b: bool = True
    ) -> None:
        """
        Set ablation mode for experiments.
        
        Args:
            enable_agent_3: Enable Tuner (False for Experiment A)
            enable_group_b: Enable Council (False for Experiment B)
        """
        self.config.ENABLE_AGENT_3 = enable_agent_3
        self.config.ENABLE_GROUP_B = enable_group_b
        
        self.logger.info(
            f"Ablation mode set: Agent3={enable_agent_3}, GroupB={enable_group_b}"
        )
