"""
Agent 2: The Inspector (Strategy Translator)

When Council majority is already FAIL, Inspector determines
the best Tuner parameters.
"""

from typing import Optional, Dict, Any, List
import time
from .base import LLMAgent, AgentResult, AgentType, Decision, CaseFile
from ..config import ACAConfig


INSPECTOR_SYSTEM_PROMPT_UUV = """You are the Inspector Agent in a UUV (underwater vehicle) anomaly detection system.

The Council majority decision is ALREADY FAIL.
Your only job is to determine the best parameters for Agent 3 (Tuner).

Review the council reasoning, error, bound, uncertainty, and deviation ratio holistically.
Infer seriousness yourself from the evidence. Do not output a severity label.

Choose ONE action:
1. RECALIBRATE - Best when the evidence suggests transient disturbance, noise, or a confidence-bound issue
2. FINE_TUNE - Best when the evidence suggests persistent drift, bias, or a real system/model shift
3. TRY_BOTH - Best when the evidence is mixed or uncertain

Parameter bounds:
- `new_alpha` must be between 0.01 and 0.10
- `epochs` must be an integer between 50 and 200
- `learning_rate` must be between 0.00001 and 0.001

Output rules:
- If action is RECALIBRATE, provide `new_alpha` and set `epochs` and `learning_rate` to null
- If action is FINE_TUNE, provide `epochs` and `learning_rate` and set `new_alpha` to null
- If action is TRY_BOTH, provide all three values
- Keep reasoning short and operational

CRITICAL INSTRUCTIONS:
1. You must respond ONLY with a valid, raw JSON object.
2. DO NOT include any introductory or concluding text.
3. DO NOT wrap the output in markdown blocks (e.g., no ```json).
4. Use double quotes for all keys and string values.
5. `pass_votes`, `fail_votes`, and `epochs` must be integers.
6. `new_alpha` and `learning_rate` must be numeric decimals or null.
7. Do not use words for numbers, comments, trailing commas, or extra fields.

Respond with JSON:
{
    "majority_decision": "FAIL",
    "pass_votes": int,
    "fail_votes": int,
    "action": "RECALIBRATE" | "FINE_TUNE" | "TRY_BOTH",
    "new_alpha": float | null,
    "epochs": int | null,
    "learning_rate": float | null,
    "reasoning": "Brief explanation of vote tally and parameter choice"
}"""


class InspectorAgent(LLMAgent):
    """
    The Inspector: Strategy Translator

    Role:
    - Determine Tuner parameters after Council majority FAIL.

    Input: Three agent result JSONs from council.
    Output:
    - AgentResult with action + tuner parameters in payload.
    """

    MIN_ALPHA = 0.01
    MAX_ALPHA = 0.10
    MIN_EPOCHS = 50
    MAX_EPOCHS = 200
    MIN_LR = 1e-5
    MAX_LR = 1e-3

    def __init__(
        self,
        llm_client,
        config: Optional[ACAConfig] = None
    ):
        self.config = config or ACAConfig()
        super().__init__(
            AgentType.INSPECTOR,
            "Inspector",
            llm_client,
            INSPECTOR_SYSTEM_PROMPT_UUV
        )

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _normalize_action(self, action: Any) -> Decision:
        action_text = str(action or "").upper()
        if "RECALIB" in action_text:
            return Decision.RECALIBRATE
        if "FINE" in action_text or "TUNE" in action_text:
            return Decision.FINE_TUNE
        return Decision.TRY_BOTH

    def execute(
        self,
        case_file: CaseFile,
        **kwargs
    ) -> AgentResult:
        """
        Determine tuner action after Council majority FAIL.

        Expected kwargs:
            council_votes: list of 3 AgentResult from A4, A5, A6.

        Returns:
            AgentResult with action + tuner parameters.
        """
        t0 = time.perf_counter()

        council_votes: List[AgentResult] = kwargs.get('council_votes', [])
        pass_count = int(kwargs.get('pass_votes', sum(1 for v in council_votes if v.decision == Decision.PASS)))
        fail_count = int(kwargs.get('fail_votes', len(council_votes) - pass_count))

        # Build vote summary for logging / LLM context
        vote_details = []
        for v in council_votes:
            vote_details.append({
                'agent': v.agent_type.value,
                'vote': v.decision.value,
                'confidence': v.confidence,
                'reasoning': v.reasoning,
                'risk_level': v.payload.get('risk_level'),
            })

        # Majority FAIL is already decided upstream by orchestrator.
        context = {
            'sample_id': case_file.sample_id,
            'error': case_file.error,
            'bound': case_file.bound,
            'uncertainty': case_file.uncertainty,
            'deviation_ratio': case_file.error / case_file.bound if case_file.bound > 0 else float('inf'),
            'council_votes': vote_details,
            'pass_votes': pass_count,
            'fail_votes': fail_count,
            'majority_decision': 'FAIL',
        }

        llm_response = self.query_llm(context)

        decision = self._normalize_action(llm_response.get('action', 'TRY_BOTH'))

        default_alpha = self._clamp(
            self._coerce_float(self.config.recalibration_alpha, 0.05),
            self.MIN_ALPHA,
            self.MAX_ALPHA,
        )
        default_epochs = self._coerce_int(self.config.fine_tune_epochs, self.MIN_EPOCHS)
        default_epochs = int(self._clamp(default_epochs, self.MIN_EPOCHS, self.MAX_EPOCHS))
        default_lr = self._clamp(
            self._coerce_float(self.config.fine_tune_lr, 1e-4),
            self.MIN_LR,
            self.MAX_LR,
        )

        # Build instruction payload for Tuner
        payload = {
            'majority_decision': 'FAIL',
            'pass_votes': pass_count,
            'fail_votes': fail_count,
            'vote_details': vote_details,
            'action': decision.value,
            'reasoning': llm_response.get('reasoning', 'Majority FAIL, applying default action.'),
        }

        if decision in (Decision.RECALIBRATE, Decision.TRY_BOTH):
            payload['new_alpha'] = self._clamp(
                self._coerce_float(
                    llm_response.get('new_alpha', default_alpha),
                    default_alpha,
                ),
                self.MIN_ALPHA,
                self.MAX_ALPHA,
            )

        if decision in (Decision.FINE_TUNE, Decision.TRY_BOTH):
            payload['epochs'] = int(self._clamp(
                self._coerce_int(
                    llm_response.get('epochs', default_epochs),
                    default_epochs,
                ),
                self.MIN_EPOCHS,
                self.MAX_EPOCHS,
            ))
            payload['learning_rate'] = self._clamp(
                self._coerce_float(
                    llm_response.get('learning_rate', default_lr),
                    default_lr,
                ),
                self.MIN_LR,
                self.MAX_LR,
            )

        result = AgentResult(
            agent_type=self.agent_type,
            decision=decision,
            confidence=self._clamp(
                self._coerce_float(llm_response.get('confidence', 0.8), 0.8),
                0.0,
                1.0,
            ),
            reasoning=payload['reasoning'],
            payload=payload
        )

        elapsed = time.perf_counter() - t0
        print(
            f"  [INSPECTOR] ({elapsed:.3f} s) "
            f"Council majority FAIL ({fail_count}/{len(council_votes)}). "
            f"Action: {decision.value}"
        )
        if 'new_alpha' in payload:
            print(f"    new_alpha: {payload['new_alpha']}")
        if 'epochs' in payload:
            print(f"    epochs: {payload['epochs']}, lr: {payload.get('learning_rate', 'N/A')}")

        case_file.add_result(result)
        self.log_execution(case_file, result)

        return result
