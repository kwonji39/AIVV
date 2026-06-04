"""Agent 6: The System Engineer

LSTM-aware council voter with deep knowledge of the model architecture,
MC Dropout uncertainty estimation, conformal prediction calibration,
and known detection blind spots.
"""

from typing import Optional, Dict, Any
import time
from .base import LLMAgent, AgentResult, AgentType, Decision, CaseFile
from ..config import ACAConfig


SYSTEM_ENGINEER_PROMPT_UUV = """You are Agent 6, the System Engineer, in a UUV anomaly detection Council.
 
You are the ONLY council member with deep knowledge of the gain-tuning of the unmanned underwater vehicle.
You provide gain-tuning proposal for the autopilot system.
You provide one equal vote (1/3) alongside Agent 4 (Requirements Engineer) and Agent 5 (Failure Manager).
 
=== CURRENT SIMULINK MODEL PARAMETERS ===
The following are the current baseline parameters for your reference:
Heading Autopilot (PID):
  - Nomoto Time Constant (T): 5
  - Nomoto Gain (K): 1
  - Proportional Gain (Kp): 0.5
  - Derivative Time (Td): 1
  - Integral Time (Ti): 20
Reference Model (3rd-order):
  - Max Velocity: 10
  - Relative Damping Ratio: 1
  - Natural Frequency: 0.2

=== VOTING RULES ===

You receive `failure_manager_findings` in the context. This is the most important signal for your vote.
You also receive `requirements_engineer_findings` with any requirement violations detected.

1. LSTM anomaly detection logic:
   - When the true data is outside the confidence bound, it is a fault for default algorithm. Vote FAIL.
   - In special case, when the true data is outside the confidence bound with significantly drifting uncertainty, it means the maneuvering is abruptly changed. Vote PASS.
   - LOW uncertainty + large error = confident fault. Vote FAIL.
   - You must discriminate between maneuver and fault damping. 

2. Confidence Interval (Bound) Width:
   - NARROW bound + true data near boundary: Vote FAIL.
   - WIDE bound + borderline error: Vote based on discrimination between maneuver and fault damping.

3. LSTM vulnerability:
   - If LSTM uncertainty is high, it is likely unexpected behavior which is not trained in the training data. VOTE FAIL.
   - If LSTM prediction is poor, it is likely unexpected behavior which is not trained in the training data. VOTE FAIL.
   - If LSTM prediction is stale with large uncertainty, VOTE FAIL.

CRITICAL INSTRUCTIONS:
1. You must respond ONLY with a valid, raw JSON object.
2. DO NOT include any introductory or concluding text.
3. DO NOT wrap the output in markdown blocks (e.g., no ```json).
4. DO NOT output your reasoning process outside the JSON.
5. ALWAYS check `failure_manager_findings` first before making your vote decision.
6. Check `requirements_engineer_findings` for any requirement violations.
7. If `failure_manager_vote` is FAIL: you should also vote FAIL (unless there is overwhelming evidence it is a false alarm).
8. If `training_evidence_summary` is provided, cite concrete comparisons (error vs expected limits, uncertainty vs expected limits).
9. Avoid keeping parameters identical when `failure_manager_vote` is FAIL — the system needs adjustment.

=== GAIN TUNING PROTOCOL (Always Required) ===
You MUST ALWAYS include a `tuning_proposal` object in your JSON response.

Strict Rule for Gain-Tuning: Only propose adjusted parameters if `failure_manager_vote` is FAIL OR `requirements_engineer_vote` is FAIL.
- If BOTH are PASS: keep ALL parameters identical to `current_gains`. Do NOT change any value.
- If `failure_manager_vote` is FAIL: propose adjusted parameters based on failure type, you must monitor the prediction data based on window.
  1) We define the post-fault window as the time window after the fault is detected.
  2) You must monitor the prediction data 'trajectory' and 'uncertainty' in the post-fault window.
  3) If the trajectory is stale with high uncertainty, increase Kp and Td.
  4) If the trajectory is oscillating with low frequency, decrease Kp and increase Ti.
  5) If the trajectory is oscillating with high frequency, it is just noise, so do not change any parameters.
  6) If the trajectory diverges from the original trajectory, decrease Td and increase Ti.
  7) If the trajectory converges to the original trajectory, it is good failure mode, so do not change any parameters.

- If `requirements_engineer_vote` is FAIL: propose parameters to bring behavior back within the violated requirement limit.

Always reference `current_gains` values as the starting point. Show the numeric direction of each change.

Return ONLY this JSON:
{
      "vote": "PASS" | "FAIL",
      "risk_level": "LOW" | "MEDIUM" | "HIGH",
      "confidence": float,
      "technical_assessment": "Brief assessment",
      "reasoning": "Reference failure_manager_findings, requirements_engineer_findings, oscillation_count, is_converging, peak_deviation, and LSTM metrics",
      "tuning_proposal": {
          "Kp": float,
          "Ti": float,
          "Td": float,
          "Reference_Max_Velocity": float
      },
      "tuning_reasoning": "If FM or RE voted FAIL: state specific values changed and the reason. If both voted PASS: state 'No adjustment needed. Current gains maintained.'"
}"""


class SystemEngineerAgent(LLMAgent):

    WEIGHT = 1.0 / 3.0

    def __init__(
        self,
        llm_client,
        config: Optional[ACAConfig] = None
    ):
        self.config = config or ACAConfig()
        super().__init__(
            AgentType.SYSTEM_ENGINEER,
            "System Engineer",
            llm_client,
            SYSTEM_ENGINEER_PROMPT_UUV
        )

    def execute(
        self,
        case_file: CaseFile,
        adaptation_payload: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """Generate Agent 6 vote (PASS/FAIL) with LSTM-aware analysis."""
        t0 = time.perf_counter()

        if adaptation_payload is None:
            adaptation_payload = {}

        deviation_ratio = case_file.error / case_file.bound if case_file.bound > 0 else float('inf')
        evidence_summary = self._build_training_evidence_summary(case_file, adaptation_payload)

        # Extract Failure Manager findings from already-executed agents in the case_file
        failure_manager_findings = {}
        for r in reversed(case_file.agent_results):
            if r.agent_type == AgentType.FAILURE_MANAGER:
                fm_payload = r.payload or {}
                fm_metrics = fm_payload.get('failure_metrics', {})
                failure_manager_findings = {
                    'failure_manager_vote': r.decision.value,
                    'oscillation_count': fm_metrics.get('oscillation_count'),
                    'is_converging': fm_metrics.get('is_converging'),
                    'peak_deviation': fm_metrics.get('peak_deviation'),
                    'settling_trend': fm_metrics.get('settling_trend'),
                    'fm_assessment': fm_payload.get('failure_management_assessment', ''),
                }
                break

        # Extract Requirements Engineer findings
        requirements_engineer_findings = {}
        for r in reversed(case_file.agent_results):
            if r.agent_type == AgentType.REQUIREMENTS_ENGINEER:
                re_payload = r.payload or {}
                requirements_engineer_findings = {
                    'requirements_engineer_vote': r.decision.value,
                    'requirement_section': re_payload.get('requirement_section', ''),
                    'veto_reason': re_payload.get('veto_reason', ''),
                    'reasoning': r.reasoning or '',
                }
                break

        # Build context for LLM
        context = {
            'sample_id': case_file.sample_id,
            # Core detection metrics
            'error': case_file.error,
            'bound': case_file.bound,
            'deviation_ratio': deviation_ratio,
            'uncertainty': case_file.uncertainty,
            'alpha': case_file.alpha,
            # Failure Manager findings — PRIMARY signal for gain tuning
            'failure_manager_findings': failure_manager_findings,
            # Requirements Engineer findings — SECONDARY signal
            'requirements_engineer_findings': requirements_engineer_findings,
            # LSTM system info
            'lstm_info': {
                'mc_dropout_uncertainty': case_file.uncertainty,
                'conformal_alpha': case_file.alpha,
                'confidence_interval_width': case_file.bound,
                'error_to_ci_ratio': deviation_ratio,
                'ci_is_narrow': case_file.bound < 0.5,
                'model_is_confident': case_file.uncertainty < 0.1,
            },
            'training_evidence_summary': evidence_summary,
            # Tuner results if available
            'proposal': {
                'action_taken': adaptation_payload.get('action_taken'),
                'old_bound': adaptation_payload.get('old_bound'),
                'new_bound': adaptation_payload.get('new_bound'),
                'old_error': adaptation_payload.get('old_error'),
                'new_error': adaptation_payload.get('new_error'),
                'applied_alpha': adaptation_payload.get('applied_alpha'),
            },
            'control_metrics': {
                'max_overshoot_deg': adaptation_payload.get('overshoot'),
                'settling_time_sec': adaptation_payload.get('settling_time'),
                'steady_state_error': adaptation_payload.get('ss_error')
            },
            'current_gains': adaptation_payload.get('current_gains', {
                'Kp': 0.5, 'Td': 1.0, 'Ti': 20.0, 'Nomoto_T': 5, 'Nomoto_K': 1,
                'Reference_Max_Velocity': 10, 'Damping_Ratio': 1.0, 'Natural_Frequency': 0.2
            }),
        }

        llm_response = self.query_llm(context)

        vote = str(llm_response.get('vote', 'PASS')).upper()
        vote = 'PASS' if vote == 'PASS' else 'FAIL'
        decision = Decision.PASS if vote == 'PASS' else Decision.FAIL
        risk_level = str(llm_response.get('risk_level', 'MEDIUM')).upper()
        confidence = float(llm_response.get('confidence', 0.75))
        reasoning = llm_response.get(
            'reasoning',
            f"SystemEngineer vote={vote} | ratio={deviation_ratio:.2f}, "
            f"uncertainty={case_file.uncertainty:.4f}, bound={case_file.bound:.4f}"
        )
        technical_assessment = llm_response.get('technical_assessment', '')
        tuning_proposal = llm_response.get('tuning_proposal', None)
        tuning_reasoning = llm_response.get('tuning_reasoning', '')

        result = AgentResult(
            agent_type=self.agent_type,
            decision=decision,
            confidence=max(0.5, min(1.0, confidence)),
            reasoning=reasoning,
            payload={
                'vote': vote,
                'weight': self.WEIGHT,
                'risk_level': risk_level,
                'deviation_ratio': deviation_ratio,
                'technical_assessment': technical_assessment,
                'mc_uncertainty': case_file.uncertainty,
                'ci_width': case_file.bound,
                'training_evidence_summary': evidence_summary,
                'proposed_gains': tuning_proposal,
                'tuning_reasoning': tuning_reasoning,
            }
        )

        elapsed = time.perf_counter() - t0
        print(
            f"  [COUNCIL] ({elapsed:.3f} s) SystemEngineer vote={vote} "
            f"risk={risk_level} ratio={deviation_ratio:.2f} "
            f"unc={case_file.uncertainty:.4f}"
        )

        case_file.add_result(result)
        self.log_execution(case_file, result)

        return result

    def _build_training_evidence_summary(
        self,
        case_file: CaseFile,
        adaptation_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize current sample vs historical baseline for LLM prompt context."""
        training_evidence = adaptation_payload.get('candidate_training_evidence') or adaptation_payload.get('training_evidence', {})
        if not isinstance(training_evidence, dict):
            return {}

        train_probe = training_evidence.get('model_baseline', {}).get('train_probe', {})
        raw_baseline = training_evidence.get('raw_window_baseline', {})
        frame_stats = adaptation_payload.get('frame_stats', {})

        train_error_max = float(train_probe.get('error_distribution', {}).get('limit', 0.0) or 0.0)
        train_sigma_max = float(train_probe.get('uncertainty_distribution', {}).get('limit', 0.0) or 0.0)
        step_max = float(raw_baseline.get('window_max_abs_step_distribution', {}).get('limit', 0.0) or 0.0)
        range_max = float(raw_baseline.get('window_range_distribution', {}).get('limit', 0.0) or 0.0)

        current_error = float(adaptation_payload.get('new_error', adaptation_payload.get('error', case_file.error)) or case_file.error or 0.0)
        current_uncertainty = float(adaptation_payload.get('new_uncertainty', adaptation_payload.get('uncertainty', case_file.uncertainty)) or case_file.uncertainty or 0.0)
        current_step = float(frame_stats.get('max_abs_step_delta', 0.0) or 0.0)
        current_range = float(frame_stats.get('range', 0.0) or 0.0)

        evidence_balance = []
        if current_error > train_error_max and train_error_max > 0:
            evidence_balance.append('error_exceeds_max')
        if current_uncertainty > train_sigma_max and train_sigma_max > 0:
            evidence_balance.append('uncertainty_exceeds_max')
        if current_step > step_max and step_max > 0:
            evidence_balance.append('step_exceeds_max')
        if current_range > range_max and range_max > 0:
            evidence_balance.append('range_exceeds_max')

        return {
            'train_error_max': train_error_max,
            'train_sigma_max': train_sigma_max,
            'current_error': current_error,
            'current_uncertainty': current_uncertainty,
            'current_step': current_step,
            'current_range': current_range,
            'step_max': step_max,
            'range_max': range_max,
            'elevated_metrics': evidence_balance,
        }

