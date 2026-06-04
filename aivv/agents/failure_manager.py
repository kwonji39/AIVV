"""
Agent 5: The Failure Manager (Failure Mode V&V)

Validates whether the system's response to a detected fault
satisfies failure management requirements (damping, failure effects,
settling behavior). Does NOT re-classify fault type -- instead
serves as the independent runtime V&V agent for the off-nominal regime.
"""

from typing import Optional, Dict, Any, List
from .base import LLMAgent, AgentResult, AgentType, Decision, CaseFile
from ..config import ACAConfig


FAILURE_MANAGER_PROMPT_UUV = """You are Agent 5, the Failure Manager, in a UUV anomaly detection Council.

Your role is SYSTEM VALIDATION in FAILURE MODE.
Sentry has flagged a potential fault. Your job is NOT to re-check whether there is a fault.
Your job is to answer: "Given that a fault may be present, is the system's response within failure management requirements?"

You are the V&V agent for the off-nominal (failure) regime.

=== YOUR V&V RESPONSIBILITIES ===

You must evaluate the system's response using the provided `frame_stats`, raw `frame_values`, and `failure_management_requirements`.
You evaluate whether the failure effect is acceptable and whether the system is appropriately recovering or handling the failure based on the natural language requirements provided.

1. Failure Effect:
   - Does the maximum deviation (e.g., `frame_stats.max_abs_value`) exceed the limits specified in the natural language requirements?
   
2. Recovery and Damping:
   - Examine the sequence in `frame_values`. Is the response converging (settling down) or diverging (getting worse)?
   - If the system is diverging or oscillating uncontrollably, that is a failure management concern.

3. Baseline Context:
   - Use `frame_baseline_summary` to compare against clean-training limits.
   - If the current deviation is within expected normal limits (`abs_value_max`), the effect is likely contained.

=== VOTING LOGIC ===

Before voting, you must check both FAIL and PASS conditions. Never vote without checking both conditions.

Vote FAIL when:
- If the true data is outside the confidence bound with significant gradient change, Vote FAIL.
- If the true data is outside the confidence bound and current predicted point is significantly different (higher than noise level) from the previous point, Vote FAIL.
- If the predicted data is significantly outside the bound which is learned during training, Vote FAIL.

Vote PASS when:
- If the true data is outside the confidence bound but the predicted data is significantly different from the true data, it is a prediction issue. Vote PASS.
- The failure effect is contained and the system is recovering properly.
- If the predicted data is not involving any failure effect, it is a normal operation. Vote PASS.
- If the predicted data is not involving any damping or recovery, it is a noise. Vote PASS.

CRITICAL INSTRUCTIONS:
1. You must respond ONLY with a valid, raw JSON object.
2. DO NOT include any introductory or concluding text.
3. DO NOT wrap the output in markdown blocks.
4. DO NOT output your reasoning process outside the JSON.
5. The `confidence` field MUST be a numeric decimal such as 0.72. Never use words.
6. Your role is FAILURE MODE V&V -- assess the system response, not the fault cause.

Return ONLY this JSON:
{
    "vote": "PASS" | "FAIL",
    "risk_level": "LOW" | "MEDIUM" | "HIGH",
    "confidence": float,
    "failure_management_assessment": "Which natural language requirement was evaluated and result",
    "reasoning": "Explanation based on values and natural language requirement thresholds"
}"""


def _compute_failure_metrics(frame_values: List[float], frame_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Derive failure management V&V metrics from the current window."""
    if not frame_values or len(frame_values) < 4:
        return {
            'peak_deviation': float(frame_stats.get('max_abs_value', 0.0)),
            'is_converging': None,
            'settling_trend': 0.0,
            'oscillation_count': 0,
            'first_half_mean_abs': None,
            'second_half_mean_abs': None,
        }

    n = len(frame_values)
    peak_deviation = float(frame_stats.get('max_abs_value', max(abs(v) for v in frame_values)))
    last_value = float(frame_stats.get('last_value', frame_values[-1]))

    # Settling trend: negative = last abs value < peak (converging), positive = diverging
    settling_trend = abs(last_value) - peak_deviation

    # Convergence check: compare mean abs value of second half vs first half of window
    mid = n // 2
    first_half_abs = [abs(v) for v in frame_values[:mid]]
    second_half_abs = [abs(v) for v in frame_values[mid:]]
    first_half_mean = sum(first_half_abs) / len(first_half_abs) if first_half_abs else 0.0
    second_half_mean = sum(second_half_abs) / len(second_half_abs) if second_half_abs else 0.0
    is_converging = second_half_mean < first_half_mean

    # Oscillation count: number of sign changes in consecutive deltas
    deltas = [frame_values[i+1] - frame_values[i] for i in range(n - 1)]
    oscillation_count = sum(
        1 for i in range(len(deltas) - 1)
        if deltas[i] * deltas[i + 1] < 0  # sign flip = direction reversal
    )

    return {
        'peak_deviation': peak_deviation,
        'is_converging': is_converging,
        'settling_trend': settling_trend,
        'oscillation_count': oscillation_count,
        'first_half_mean_abs': first_half_mean,
        'second_half_mean_abs': second_half_mean,
        'interpretation': (
            'CONVERGING (good damping)'
            if is_converging
            else 'DIVERGING (damping concern)'
        ),
    }


def _build_frame_baseline_summary(training_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Extract frame-level baseline limits for the Failure Manager."""
    if not isinstance(training_evidence, dict):
        return {}

    raw_window_baseline = training_evidence.get('raw_window_baseline', {})
    return {
        'abs_value_q99': raw_window_baseline.get('abs_value_distribution', {}).get('q99'),
        'range_q99': raw_window_baseline.get('window_range_distribution', {}).get('q99'),
        'step_q99': raw_window_baseline.get('window_max_abs_step_distribution', {}).get('q99'),
    }


def _build_failure_management_requirements(
    frame_baseline_summary: Dict[str, Any],
    oscillation_fail_threshold: int = 6,
    failure_effect_multiplier: float = 1.5,
) -> Dict[str, Any]:
    """Derive concrete failure management requirement thresholds from training baseline."""
    abs_q99 = float(frame_baseline_summary.get('abs_value_q99') or 80.0)
    range_q99 = float(frame_baseline_summary.get('range_q99') or 40.0)

    return {
        'max_failure_effect': abs_q99 * failure_effect_multiplier,
        'max_allowable_range': range_q99 * failure_effect_multiplier,
        'oscillation_fail_threshold': oscillation_fail_threshold,
        'required_response': 'CONVERGING (is_converging=True or settling_trend <= 0)',
        'description': (
            f"Failure limits: peak_deviation<={abs_q99 * failure_effect_multiplier:.2f}, "
            f"range<={range_q99 * failure_effect_multiplier:.2f}, "
            f"oscillation_count<={oscillation_fail_threshold}, "
            f"response must be CONVERGING"
        ),
    }


class FailureManagerAgent(LLMAgent):
    """
    Agent 5: The Failure Manager (Failure Mode V&V)

    Role: Verify the system's response to a detected fault satisfies failure management requirements.
    Method: Physics-driven metric computation (damping, failure effect, oscillation) + LLM V&V.
    Weight: 1/3 (equal council vote).

    V&V Question: "Is the system recovering properly from the detected fault?"
    Output: VOTE PASS (failure effect contained, good recovery) or VOTE FAIL (requirement violated).
    """

    WEIGHT = 1.0 / 3.0
    # Failure management thresholds (can be overridden via config)
    OSCILLATION_FAIL_THRESHOLD = 6
    FAILURE_EFFECT_MULTIPLIER = 1.5

    def __init__(
        self,
        llm_client,
        config: Optional[ACAConfig] = None,
    ):
        self.config = config or ACAConfig()
        super().__init__(
            AgentType.FAILURE_MANAGER,
            "Failure Manager",
            llm_client,
            FAILURE_MANAGER_PROMPT_UUV
        )

    def execute(
        self,
        case_file: CaseFile,
        adaptation_payload: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """
        Verify failure management requirements (Failure Mode V&V).

        Args:
            case_file: Case file with context
            adaptation_payload: Payload from orchestrator council context

        Returns:
            AgentResult with PASS or FAIL vote
        """
        if adaptation_payload is None:
            for result in reversed(case_file.agent_results):
                if result.agent_type == AgentType.TUNER:
                    adaptation_payload = result.payload.get('adaptation_payload', {})
                    break
            if adaptation_payload is None:
                adaptation_payload = {}

        deviation_ratio = case_file.error / case_file.bound if case_file.bound > 0 else float('inf')
        frame_values = adaptation_payload.get('frame_values', [])
        frame_stats = adaptation_payload.get('frame_stats', {})
        training_evidence = (
            adaptation_payload.get('candidate_training_evidence')
            or adaptation_payload.get('training_evidence', {})
        )

        # Compute V&V metrics
        failure_metrics = _compute_failure_metrics(frame_values, frame_stats)
        frame_baseline_summary = _build_frame_baseline_summary(training_evidence)
        failure_management_requirements = _build_failure_management_requirements(
            frame_baseline_summary,
            oscillation_fail_threshold=self.OSCILLATION_FAIL_THRESHOLD,
            failure_effect_multiplier=self.FAILURE_EFFECT_MULTIPLIER,
        )

        # Deterministic quick check: clear failure management violation
        quick_result = self._quick_check(
            case_file, deviation_ratio,
            failure_metrics, failure_management_requirements,
            frame_stats, frame_baseline_summary,
        )
        if quick_result is not None:
            return quick_result

        # Full LLM-based V&V assessment
        context = {
            'sample_id': case_file.sample_id,
            'v_and_v_mode': 'FAILURE_MODE',
            'system_mode': adaptation_payload.get('system_mode', 'FAULT_SUSPECT'),
            'deviation_ratio': deviation_ratio,
            'bound': case_file.bound,
            'uncertainty': case_file.uncertainty,
            'failure_metrics': failure_metrics,
            'failure_management_requirements': failure_management_requirements,
            'frame_stats': frame_stats,
            'frame_baseline_summary': frame_baseline_summary,
            'proposal': {
                'action_taken': adaptation_payload.get('action_taken'),
                'old_bound': adaptation_payload.get('old_bound'),
                'new_bound': adaptation_payload.get('new_bound'),
            },
        }

        llm_response = self.query_llm(context)

        vote = str(llm_response.get('vote', 'FAIL')).upper()
        vote = 'PASS' if vote == 'PASS' else 'FAIL'
        risk_level = str(llm_response.get('risk_level', 'MEDIUM')).upper()
        confidence = float(llm_response.get('confidence', 0.6))
        reasoning = llm_response.get(
            'reasoning',
            f"FailureManager V&V: peak={failure_metrics['peak_deviation']:.2f}, "
            f"converging={failure_metrics['is_converging']}, "
            f"oscillations={failure_metrics['oscillation_count']}"
        )
        fm_assessment = llm_response.get('failure_management_assessment', '')

        decision = Decision.PASS if vote == 'PASS' else Decision.FAIL

        result = AgentResult(
            agent_type=self.agent_type,
            decision=decision,
            confidence=max(0.5, min(1.0, confidence)),
            reasoning=reasoning,
            payload={
                'vote': vote,
                'weight': self.WEIGHT,
                'v_and_v_mode': 'FAILURE_MODE',
                'risk_level': risk_level,
                'deviation_ratio': deviation_ratio,
                'failure_metrics': failure_metrics,
                'failure_management_requirements': failure_management_requirements,
                'failure_management_assessment': fm_assessment,
                'quick_check': False,
            }
        )

        case_file.add_result(result)
        self.log_execution(case_file, result)

        return result

    def _quick_check(
        self,
        case_file: CaseFile,
        deviation_ratio: float,
        failure_metrics: Dict[str, Any],
        failure_management_requirements: Dict[str, Any],
        frame_stats: Dict[str, Any],
        frame_baseline_summary: Dict[str, Any],
    ) -> Optional[AgentResult]:
        """
        Deterministic V&V quick check for clear failure management violations.
        Returns an AgentResult immediately if a hard requirement is violated,
        or None to proceed to LLM assessment.
        """
        if not frame_stats or deviation_ratio <= 1.0:
            return None

        peak_deviation = failure_metrics.get('peak_deviation', 0.0)
        is_converging = failure_metrics.get('is_converging', True)
        oscillation_count = failure_metrics.get('oscillation_count', 0)
        settling_trend = failure_metrics.get('settling_trend', 0.0)
        max_failure_effect = float(failure_management_requirements.get('max_failure_effect', float('inf')))
        oscillation_threshold = int(failure_management_requirements.get('oscillation_fail_threshold', 6))

        hard_violation = False
        violation_reason = []

        if peak_deviation > max_failure_effect:
            hard_violation = True
            violation_reason.append(
                f"peak_deviation={peak_deviation:.2f} exceeds max_failure_effect={max_failure_effect:.2f}"
            )

        if (not is_converging) and settling_trend > 0 and peak_deviation > float(frame_baseline_summary.get('abs_value_q99') or 80.0):
            hard_violation = True
            violation_reason.append(
                f"DIVERGING response (settling_trend={settling_trend:.2f}) with elevated peak above training q99"
            )

        if oscillation_count > oscillation_threshold:
            hard_violation = True
            violation_reason.append(
                f"oscillation_count={oscillation_count} exceeds threshold={oscillation_threshold} (underdamped)"
            )

        if not hard_violation:
            return None

        result = AgentResult(
            agent_type=self.agent_type,
            decision=Decision.FAIL,
            confidence=0.92,
            reasoning=(
                "Failure mode V&V quick check: failure management requirement violated. "
                + "; ".join(violation_reason)
            ),
            payload={
                'vote': 'FAIL',
                'weight': self.WEIGHT,
                'v_and_v_mode': 'FAILURE_MODE',
                'risk_level': 'HIGH',
                'deviation_ratio': deviation_ratio,
                'failure_metrics': failure_metrics,
                'failure_management_requirements': failure_management_requirements,
                'failure_management_assessment': '; '.join(violation_reason),
                'quick_check': True,
            }
        )

        case_file.add_result(result)
        self.log_execution(case_file, result)

        return result
