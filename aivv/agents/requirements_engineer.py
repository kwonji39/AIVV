"""
Agent 4: The Requirements Engineer (Normal Mode V&V)

Validates whether the system satisfies operational requirements
during nominal operation. Does NOT replicate Sentry's anomaly
detection logic -- instead serves as the independent runtime
Verification & Validation agent for the nominal regime.
"""

from typing import Optional, Dict, Any
from .base import LLMAgent, AgentResult, AgentType, Decision, CaseFile
from ..config import ACAConfig


REQUIREMENTS_ENGINEER_PROMPT_UUV = """You are Agent 4, the Requirements Engineer, in a UUV anomaly detection Council.

Your role is SYSTEM VALIDATION in NORMAL MODE.
You verify whether the system satisfies its operational requirements, independent of whether Sentry flagged an anomaly.

You are NOT re-checking Sentry's fault detection. You are answering a different question:
"Even assuming a potential fault exists, does the system's current behavior violate any operational requirement?"

=== YOUR V&V RESPONSIBILITIES ===

You must evaluate the system's response using the provided `frame_stats`, raw `frame_values`, and `operational_requirements`.
You evaluate whether the normal operational requirements are satisfied or violated based on the natural language requirements provided.

1. Normal Operation:
   - Yaw angle (rate) has entire operational range of -180 to 180.
   - Yaw angle (rate) has per-step operational range of -10 to 10.
   - Yaw angle (rate) has environmental / sensor drift noise
   - The UUV is controlled with sampling rate 0.05s.

2. Operational Limits:
   - Although the predicted value involves significant damping, it must not exceed normal operational range.
   - Compare the current values to the training limits to ensure the system is operating within a normal envelope.
   - Damping level must not exceed normal operational range. (per-step)
   - After damping, the predicted value must not exceed normal operational range. (per-step)

3. Sensor Noise Discrimination:
   - Sensor noise is normal when the step size is within the normal operational range.
   - Sensor noise is abnormal when the step size is outside the normal operational range.
   - If the step size is within the normal operational range, it is not a fault.
   - If the step size is outside the normal operational range, it is a fault.

4. Slow Drift Requirement (Masking Risk):
   - The confidence bound multiplier MUST NEVER exceed the specified limit (typically 2.0 times the original bound).
   - If `bound_multiplier` > 2.0: FAIL (masking risk, policy violation).

=== VOTING LOGIC ===

Before voting, you must check both FAIL and PASS conditions. Never vote without checking both conditions.

Vote FAIL when:
- Your default vote is FAIL but you have to check PASS conditions.
- The predicted value is significantly different from the previous point (more than noise level, over the requirement level). Vote FAIL.
- If the predicted data is stale (exactly same as previous point), it is a failure in electrical system. Vote FAIL.

Vote PASS only when (the others are FAIL):
- Although the true valus is outside the confidence bounds, the predicted value is in the per-step operational range. Vote PASS.
- If the current predicted value is within the noise level. Vote PASS.
- If the current predicted value is within the confidence bound and the step size is within the noise level. Vote PASS.
- If the predicted data is in the similar range of the previous few points, it can be a noise, Vote PASS.

CRITICAL INSTRUCTIONS:
1. You must respond ONLY with a valid, raw JSON object.
2. DO NOT include any introductory or concluding text.
3. DO NOT wrap the output in markdown blocks.
4. DO NOT output your reasoning process outside the JSON.
5. Cite specific numeric values from `training_comparison` in your reasoning.
6. Identify PASS-supporting evidence AND FAIL-supporting evidence before deciding.
7. Your role is NORMAL MODE V&V -- do not duplicate Sentry's fault detection logic.

Return ONLY this JSON:
{
    "vote": "PASS" | "FAIL",
    "confidence": float,
    "requirement_section": "Which natural language requirement was evaluated",
    "reasoning": "Explanation citing evaluated values and natural language limits",
    "veto_reason": "Specific requirement violated" | null
}"""


# UUV Operational Requirements Knowledge Base
UUV_OPERATIONAL_REQUIREMENTS_KB = {
    "max_bound_multiplier": 2.0,
    "ratio_fail_threshold": 1.5,
    "multi_signal_fail_count": 2,
    "high_noise_sensors": ["yaw", "imu", "depth", "rate", "flow"],
    "slow_drift_window": 3,
    "alpha_limits": {
        "min": 0.01,
        "high_noise_min": 0.02,
        "critical_min": 0.05,
    },
}


class RequirementsEngineerAgent(LLMAgent):
    """
    Agent 4: The Requirements Engineer (Normal Mode V&V)

    Role: Verify the system satisfies operational requirements during nominal operation.
    Method: Ratio-based comparison of current system state vs. training baseline.
    Weight: 1/3 (equal council vote).

    V&V Question: "Does the current system behavior violate any operational requirement?"
    Output: VOTE PASS (requirements satisfied) or VOTE FAIL (requirement violated).
    """

    WEIGHT = 1.0 / 3.0

    def __init__(
        self,
        llm_client,
        config: Optional[ACAConfig] = None,
        knowledge_base: Optional[Dict] = None
    ):
        self.config = config or ACAConfig()
        super().__init__(
            AgentType.REQUIREMENTS_ENGINEER,
            "Requirements Engineer",
            llm_client,
            REQUIREMENTS_ENGINEER_PROMPT_UUV
        )
        self.kb = knowledge_base or UUV_OPERATIONAL_REQUIREMENTS_KB

    def execute(
        self,
        case_file: CaseFile,
        adaptation_payload: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AgentResult:
        """
        Verify operational requirements compliance (Normal Mode V&V).

        Args:
            case_file: Case file with context
            adaptation_payload: Payload from the tuner / orchestrator council context

        Returns:
            AgentResult with PASS or FAIL vote
        """
        if adaptation_payload is None:
            tuner_result = self._get_tuner_result(case_file)
            if tuner_result and 'adaptation_payload' in tuner_result.payload:
                adaptation_payload = tuner_result.payload['adaptation_payload']
            else:
                adaptation_payload = {}

        training_comparison = self._build_training_comparison(case_file, adaptation_payload)
        operational_requirements = self._build_operational_requirements(training_comparison)

        context = {
            'sample_id': case_file.sample_id,
            'v_and_v_mode': 'NORMAL_MODE',
            'system_mode': adaptation_payload.get('system_mode', 'FAULT_SUSPECT'),
            'proposal': {
                'action_taken': adaptation_payload.get('action_taken'),
                'old_bound': adaptation_payload.get('old_bound'),
                'new_bound': adaptation_payload.get('new_bound'),
                'old_error': adaptation_payload.get('old_error'),
                'new_error': adaptation_payload.get('new_error'),
                'applied_alpha': adaptation_payload.get('applied_alpha'),
            },
            'original_alpha': case_file.alpha,
            'original_bound': case_file.bound,
            'error_magnitude': case_file.error,
            'training_comparison': training_comparison,
            'operational_requirements': operational_requirements,
            'process_rules': self.kb,
        }

        llm_response = self.query_llm(context)

        vote = str(llm_response.get('vote', 'PASS')).upper()
        vote = 'PASS' if vote == 'PASS' else 'FAIL'
        decision = Decision.PASS if vote == 'PASS' else Decision.FAIL

        result = AgentResult(
            agent_type=self.agent_type,
            decision=decision,
            confidence=float(llm_response.get('confidence', 0.8)),
            reasoning=llm_response.get('reasoning', 'Normal mode V&V completed'),
            payload={
                'vote': vote,
                'weight': self.WEIGHT,
                'v_and_v_mode': 'NORMAL_MODE',
                'requirement_section': llm_response.get('requirement_section'),
                'veto_reason': llm_response.get('veto_reason'),
                'training_comparison': training_comparison,
                'operational_requirements': operational_requirements,
            }
        )

        case_file.add_result(result)
        self.log_execution(case_file, result)

        return result

    def _get_tuner_result(self, case_file: CaseFile) -> Optional[AgentResult]:
        """Get Tuner's result from case file."""
        for result in reversed(case_file.agent_results):
            if result.agent_type == AgentType.TUNER:
                return result
        return None

    def _build_operational_requirements(self, training_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Derive concrete operational requirement thresholds from training baseline."""
        metrics = training_comparison.get('metrics', {})
        fail_threshold = self.kb.get('ratio_fail_threshold', 1.5)
        max_bound_mult = self.kb.get('max_bound_multiplier', 2.0)

        error_q99 = metrics.get('error', {}).get('train_q99', 0.0) or 0.0
        range_q99 = metrics.get('window_range', {}).get('train_q99', 0.0) or 0.0
        step_q99 = metrics.get('max_abs_step', {}).get('train_q99', 0.0) or 0.0

        return {
            'max_allowable_error': error_q99 * fail_threshold,
            'max_allowable_range': range_q99 * fail_threshold,
            'max_allowable_step': step_q99 * fail_threshold,
            'max_bound_multiplier': max_bound_mult,
            'multi_signal_fail_count': self.kb.get('multi_signal_fail_count', 2),
            'ratio_fail_threshold': fail_threshold,
            'description': (
                f"Operational limits: error<={error_q99*fail_threshold:.4f}, "
                f"range<={range_q99*fail_threshold:.4f}, "
                f"step<={step_q99*fail_threshold:.4f}, "
                f"bound_multiplier<={max_bound_mult:.1f}"
            ),
        }

    def _build_training_comparison(self, case_file: CaseFile, adaptation_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize how the current sample compares with training-derived baseline evidence."""
        evidence = adaptation_payload.get('candidate_training_evidence') or adaptation_payload.get('training_evidence', {})
        if not isinstance(evidence, dict):
            return {}

        raw_baseline = evidence.get('raw_window_baseline', {})
        train_probe = evidence.get('model_baseline', {}).get('train_probe', {})
        uncertainty_probe = train_probe.get('uncertainty_distribution', {})
        frame_stats = adaptation_payload.get('frame_stats', {})

        current_error = float(adaptation_payload.get('new_error', adaptation_payload.get('error', case_file.error)) or case_file.error or 0.0)
        current_uncertainty = float(adaptation_payload.get('new_uncertainty', adaptation_payload.get('uncertainty', case_file.uncertainty)) or case_file.uncertainty or 0.0)
        current_bound = float(adaptation_payload.get('new_bound', adaptation_payload.get('bound', 0.0)) or 0.0)
        old_bound = float(adaptation_payload.get('old_bound', current_bound) or current_bound or 0.0)

        error_q99 = float(train_probe.get('error_distribution', {}).get('q99', 0.0) or 0.0)
        uncertainty_q95 = float(uncertainty_probe.get('q95', 0.0) or 0.0)
        range_q99 = float(raw_baseline.get('window_range_distribution', {}).get('q99', 0.0) or 0.0)
        step_q99 = float(raw_baseline.get('window_max_abs_step_distribution', {}).get('q99', 0.0) or 0.0)

        frame_range = float(frame_stats.get('range', 0.0) or 0.0)
        max_abs_step = float(frame_stats.get('max_abs_step_delta', 0.0) or 0.0)

        bound_multiplier = (current_bound / old_bound) if old_bound > 0 else float('inf')
        ratios = {
            'error_to_train_q99': (current_error / error_q99) if error_q99 > 0 else None,
            'uncertainty_to_train_q95': (current_uncertainty / uncertainty_q95) if uncertainty_q95 > 0 else None,
            'range_to_train_q99': (frame_range / range_q99) if range_q99 > 0 else None,
            'step_to_train_q99': (max_abs_step / step_q99) if step_q99 > 0 else None,
        }

        training_like = []
        elevated = []
        for name, value in ratios.items():
            if value is None:
                continue
            item = {'metric': name, 'ratio': float(value)}
            if value <= 1.0:
                training_like.append(item)
            else:
                elevated.append(item)

        all_ratios = training_like + elevated
        strongest_signal = max(all_ratios, key=lambda item: item['ratio']) if all_ratios else None

        return {
            'bound_multiplier': float(bound_multiplier),
            'metrics': {
                'error': {'current': current_error, 'train_q99': error_q99},
                'uncertainty': {'current': current_uncertainty, 'train_q95': uncertainty_q95},
                'window_range': {'current': frame_range, 'train_q99': range_q99},
                'max_abs_step': {'current': max_abs_step, 'train_q99': step_q99},
            },
            'ratios': ratios,
            'evidence_balance': {
                'supports_pass': training_like,
                'supports_fail': elevated,
                'support_counts': {
                    'pass_like': len(training_like),
                    'fail_like': len(elevated),
                },
                'strongest_signal': strongest_signal,
            },
        }
