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

Vote FAIL when:
- If the true data is outside the confidence bound with significant gradient change, Vote FAIL.
- If the true data is outside the confidence bound and current predicted point is significantly different (higher than noise level) from the previous point, Vote FAIL.
- If the predicted data is significantly outside the bound which is learned during training, Vote FAIL.

Vote PASS only when:
- If the true data is outside the confidence bound but the predicted data is significantly different from the true data, it is a prediction issue. Vote PASS.
- The failure effect is contained and the system is recovering properly.
- If the predicted data is slightly outside the bound which is learned during training, it can be a noise, Vote PASS.

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


FAILURE_MANAGEMENT_KB = {
    "requirements": [
         "The system must show a converging response to any fault.",
         "The failure effect must not exceed 1.5x the normal expected limits.",
         "The system should not oscillate uncontrollably."
    ]
}





def _build_frame_baseline_summary(training_evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Extract frame-level baseline limits for the Failure Manager."""
    if not isinstance(training_evidence, dict):
        return {}

    raw_window_baseline = training_evidence.get('raw_window_baseline', {})
    return {
        'abs_value_max': raw_window_baseline.get('abs_value_distribution', {}).get('limit'),
        'range_max': raw_window_baseline.get('window_range_distribution', {}).get('limit'),
        'step_max': raw_window_baseline.get('window_max_abs_step_distribution', {}).get('limit'),
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
        knowledge_base: Optional[Dict] = None
    ):
        self.config = config or ACAConfig()
        super().__init__(
            AgentType.FAILURE_MANAGER,
            "Failure Manager",
            llm_client,
            FAILURE_MANAGER_PROMPT_UUV
        )
        self.kb = knowledge_base or FAILURE_MANAGEMENT_KB

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

        # Determine failure management requirements from KB
        failure_management_requirements = self.kb.get('requirements', [])
        frame_baseline_summary = _build_frame_baseline_summary(training_evidence)

        # Full LLM-based V&V assessment
        context = {
            'sample_id': case_file.sample_id,
            'v_and_v_mode': 'FAILURE_MODE',
            'system_mode': adaptation_payload.get('system_mode', 'FAULT_SUSPECT'),
            'deviation_ratio': deviation_ratio,
            'bound': case_file.bound,
            'uncertainty': case_file.uncertainty,
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
            f"FailureManager V&V fallback: peak={frame_stats.get('max_abs_value', 0.0):.2f}"
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
                'failure_management_requirements': failure_management_requirements,
                'failure_management_assessment': fm_assessment,
            }
        )

        case_file.add_result(result)
        self.log_execution(case_file, result)

        return result

