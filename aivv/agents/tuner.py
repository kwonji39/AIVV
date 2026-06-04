"""
Agent 3: The Tuner (Simulator with LLM)

Executes mathematical adjustments on a temporary engine clone.
Now uses LLM (gpt-oss) to intelligently recommend alpha values.
"""

from typing import Optional, Dict, Any, Tuple
import time
from dataclasses import dataclass
from .base import AgentResult, AgentType, Decision, CaseFile, LLMAgent
from ..engine.engine import UUVEngine
from ..config import ACAConfig


TUNER_SYSTEM_PROMPT_UUV = """You are Agent 3 (Tuner) in a UUV anomaly detection system.

Your role is to analyze prediction outcomes after fine-tuning/recalibration and recommend an operating ALPHA for conformal bounds.

OPERATING RULES:
- Alpha must be between 0.01 and 0.10.

Given:
- original error and new error
- bounds at 95%, 98%, 99%
- whether new error passes each bound

Respond with JSON:
{
    "recommended_alpha": float (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, or null),
    "reasoning": "Brief explanation",
    "would_pass_at_recommended": bool,
    "confidence": float
}"""


@dataclass
class TunerReevaluationCandidate:
    """Reevaluation candidate returned to Sentry after tuning."""
    new_prediction: float
    new_bound: float
    new_error: float
    new_uncertainty: float
    applied_alpha: Optional[float]
    passes_reevaluation: bool


@dataclass
class TunerAdaptationPayload:
    """Diagnostics payload describing the tuner adaptation attempt."""
    action_taken: str
    old_bound: float
    new_bound: float
    old_error: float
    new_error: float
    current_error: float
    old_prediction: float
    new_prediction: float
    improvement: float
    recommended_alpha: Optional[float]
    applied_alpha: Optional[float]
    llm_reasoning: str
    candidate_training_evidence: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_taken': self.action_taken,
            'old_bound': self.old_bound,
            'new_bound': self.new_bound,
            'old_error': self.old_error,
            'new_error': self.new_error,
            'current_error': self.current_error,
            'old_prediction': self.old_prediction,
            'new_prediction': self.new_prediction,
            'improvement': self.improvement,
            'recommended_alpha': self.recommended_alpha,
            'applied_alpha': self.applied_alpha,
            'llm_reasoning': self.llm_reasoning,
            'candidate_training_evidence': self.candidate_training_evidence,
        }


class TunerAgent(LLMAgent):
    """
    The Tuner: Simulator with LLM
    
    Role: The Calculator with Intelligence
    Method: LSTM Backpropagation OR Conformal Recalibration + LLM Analysis
    
    Process:
        1. Create a TEMPORARY CLONE (temp_engine)
        2. Execute Agent 2's order on temp_engine
        3. Use LLM to analyze results and recommend alpha
          4. Generate two linked outputs from the same tuning attempt:
              - Reevaluation candidate for Sentry
              - Adaptation payload for downstream reasoning/logging
    """
    
    def __init__(
        self,
        engine: UUVEngine,
        config: Optional[ACAConfig] = None,
        llm_client=None
    ):
        self.config = config or ACAConfig()
        super().__init__(
            agent_type=AgentType.TUNER,
            name="Tuner",
            llm_client=llm_client,
            system_prompt=TUNER_SYSTEM_PROMPT_UUV
        )
        self.engine = engine
        self.temp_engine: Optional[UUVEngine] = None
        self.last_reevaluation_candidate: Optional[TunerReevaluationCandidate] = None
        self.last_adaptation_payload: Optional[TunerAdaptationPayload] = None
    
    def execute(
        self,
        case_file: CaseFile,
        instruction: Dict[str, Any] = None,
        window=None,
        history_window=None,
        history_targets=None,
        comm_logger=None
    ) -> AgentResult:
        """
        Execute tuning simulation based on Inspector's instruction.
        
        Args:
            case_file: The current case being processed
            instruction: Dict from Agent 2 with action and parameters
            window: Current data window for prediction
            history_window: Historical data for fine-tuning
            history_targets: Historical targets for fine-tuning
            comm_logger: Optional AgentCommLogger for JSON logging
        """
        t0 = time.perf_counter()

        if instruction is None:
            instruction = {'action': 'TRY_BOTH', 'new_alpha': 0.05}
        
        # Create temporary engine clone
        self.temp_engine = self.engine.clone()
        
        action = instruction.get('action', 'TRY_BOTH')
        old_bound = case_file.bound
        old_prediction = case_file.prediction
        old_error = case_file.error
        
        # Execute action(s)
        if action == 'RECALIBRATE' or action == 'TRY_BOTH':
            self._execute_recalibration(instruction)
        
        if action == 'FINE_TUNE' or action == 'TRY_BOTH':
            if history_window is not None and history_targets is not None:
                self._execute_fine_tuning(
                    instruction,
                    history_window,
                    history_targets
                )
        
        # Get new prediction from temp engine
        if window is not None:
            new_prediction, new_uncertainty = self.temp_engine.predict(window)
        else:
            new_prediction = old_prediction
            new_uncertainty = case_file.uncertainty
        
        # Get bounds at multiple confidence levels
        bound_95 = self.temp_engine.get_conformal_interval(0.05)
        bound_98 = self.temp_engine.get_conformal_interval(0.02)
        bound_99 = self.temp_engine.get_conformal_interval(0.01)
        
        # Calculate new error
        new_error = abs(new_prediction - case_file.actual)
        
        # Calculate improvement
        if old_error > 0:
            improvement = (old_error - new_error) / old_error * 100
        else:
            improvement = 0.0
        
        # Use LLM to recommend ideal alpha
        recommended_alpha, llm_reasoning = self._get_llm_recommendation(
            old_error=old_error,
            new_error=new_error,
            bound_95=bound_95,
            bound_98=bound_98,
            bound_99=bound_99
        )

        # Actively apply recommended alpha on temp engine when available.
        # Fallback to instruction alpha if no recommendation.
        if recommended_alpha is not None:
            applied_alpha = float(recommended_alpha)
        else:
            applied_alpha = float(instruction.get('new_alpha', self.config.default_alpha))

        self.temp_engine.set_operating_alpha(applied_alpha)
        new_bound = self.temp_engine.get_conformal_interval(applied_alpha)

        # Check if passes reevaluation using applied alpha
        passes = new_error <= new_bound
        
        candidate_training_evidence = self.temp_engine.refresh_training_evidence(
            reason=f"tuner_{action.lower()}"
        )

        # Build reevaluation candidate for Sentry
        self.last_reevaluation_candidate = TunerReevaluationCandidate(
            new_prediction=new_prediction,
            new_bound=new_bound,
            new_error=new_error,
            new_uncertainty=new_uncertainty,
            applied_alpha=applied_alpha,
            passes_reevaluation=passes
        )
        
        # Build adaptation payload for downstream reasoning and logging
        self.last_adaptation_payload = TunerAdaptationPayload(
            action_taken=action,
            old_bound=old_bound,
            new_bound=new_bound,
            old_error=old_error,
            new_error=new_error,
            current_error=case_file.error,
            old_prediction=old_prediction,
            new_prediction=new_prediction,
            improvement=improvement,
            recommended_alpha=recommended_alpha,
            applied_alpha=applied_alpha,
            llm_reasoning=llm_reasoning,
            candidate_training_evidence=candidate_training_evidence,
        )
        
        # Build result
        result = AgentResult(
            agent_type=self.agent_type,
            decision=Decision.PASS if passes else Decision.FAIL,
            confidence=0.9 if passes else 0.5,
            reasoning=f"Action: {action}. Old error={old_error:.4f}, "
                     f"New error={new_error:.4f}. Recommended alpha={recommended_alpha}, applied alpha={applied_alpha}. "
                     f"LLM: {llm_reasoning[:100]}",
            payload={
                'reevaluation_candidate': {
                    'new_prediction': new_prediction,
                    'new_bound': new_bound,
                    'new_error': new_error,
                    'new_uncertainty': new_uncertainty,
                    'applied_alpha': applied_alpha,
                    'passes': passes
                },
                'adaptation_payload': self.last_adaptation_payload.to_dict(),
                'temp_engine_ready': True,
                'recommended_alpha': recommended_alpha,
                'applied_alpha': applied_alpha,
            }
        )
        
        # Verbose logging
        elapsed = time.perf_counter() - t0
        alpha_str = f"{recommended_alpha}" if recommended_alpha is not None else "NO_CHANGE"
        pass_95 = "PASS" if new_error <= bound_95 else "FAIL"
        pass_98 = "PASS" if new_error <= bound_98 else "FAIL"
        pass_99 = "PASS" if new_error <= bound_99 else "FAIL"
        print(
            f"  [TUNER → SENTRY] ({elapsed:.3f} s) "
            f"error={new_error:.4f}, bound={new_bound:.4f} | @95%:{pass_95} @98%:{pass_98} @99%:{pass_99} | "
            f"LLM recommends α={alpha_str}, applied α={applied_alpha:.3f}"
        )
        
        # Log to JSON if logger provided
        if comm_logger:
            comm_logger.log_communication(
                from_agent="tuner",
                to_agent="sentry",
                payload=self.last_reevaluation_candidate.__dict__
            )
        
        case_file.add_result(result)
        self.log_execution(case_file, result)
        
        return result
    
    def _get_llm_recommendation(
        self,
        old_error: float,
        new_error: float,
        bound_95: float,
        bound_98: float,
        bound_99: float
    ) -> Tuple[Optional[float], str]:
        """Use LLM to recommend the ideal alpha."""
        
        # Check pass/fail at each level
        passes_95 = new_error <= bound_95
        passes_98 = new_error <= bound_98
        passes_99 = new_error <= bound_99
        
        prompt = f"""Analyze this prediction result:
- Old error: {old_error:.4f}
- New error (after adjustment): {new_error:.4f}
- 95% confidence bound: {bound_95:.4f} - Would {"PASS" if passes_95 else "FAIL"}
- 98% confidence bound: {bound_98:.4f} - Would {"PASS" if passes_98 else "FAIL"}
- 99% confidence bound: {bound_99:.4f} - Would {"PASS" if passes_99 else "FAIL"}

What alpha should the system operate with for this sample?"""
        
        try:
            context = {
                'old_error': old_error,
                'new_error': new_error,
                'bound_95': bound_95,
                'bound_98': bound_98,
                'bound_99': bound_99,
                'passes_95': passes_95,
                'passes_98': passes_98,
                'passes_99': passes_99
            }
            response = self.query_llm(context, prompt)
            
            # Parse response
            recommended_alpha = response.get('recommended_alpha')
            reasoning = response.get('reasoning', 'No reasoning provided')
            
            return recommended_alpha, reasoning
            
        except Exception as e:
            # Fallback logic if LLM fails
            if passes_95:
                return 0.05, f"Fallback: passes at 95% (LLM error: {str(e)[:50]})"
            elif passes_98:
                return 0.02, f"Fallback: passes at 98% (LLM error: {str(e)[:50]})"
            elif passes_99:
                return 0.01, f"Fallback: passes at 99% (LLM error: {str(e)[:50]})"
            else:
                return None, f"Fallback: fails all levels (LLM error: {str(e)[:50]})"
    
    def _execute_recalibration(self, instruction: Dict[str, Any]) -> None:
        """Execute recalibration on temp engine."""
        new_alpha = instruction.get('new_alpha', self.config.recalibration_alpha)
        print(f"    [TUNER] Recalibrating with alpha={new_alpha}")
        self.temp_engine.recalibrate_conformal(new_alpha=new_alpha)
    
    def _execute_fine_tuning(
        self,
        instruction: Dict[str, Any],
        history_window,
        history_targets
    ) -> None:
        """Execute fine-tuning on temp engine."""
        epochs = instruction.get('epochs', self.config.fine_tune_epochs)
        lr = instruction.get('learning_rate', self.config.fine_tune_lr)
        
        print(f"    [TUNER] Fine-tuning LSTM for {epochs} epochs with lr={lr}")
        self.temp_engine.fine_tune(
            history_window,
            history_targets,
            lr=lr,
            epochs=epochs,
            verbose=True
        )
    
    def get_reevaluation_candidate(self) -> Optional[TunerReevaluationCandidate]:
        """Get the latest reevaluation candidate for Sentry."""
        return self.last_reevaluation_candidate
    
    def get_adaptation_payload(self) -> Optional[TunerAdaptationPayload]:
        """Get the latest diagnostics payload for the tuner adaptation."""
        return self.last_adaptation_payload
    
    def promote_temp_engine(self) -> None:
        """Promote temp engine changes to main engine (if approved by Council)."""
        if self.temp_engine is not None:
            self.engine.promote_from(self.temp_engine)
            print("    [TUNER] Temp engine promoted to main engine")
    
    def discard_temp_engine(self) -> None:
        """Discard temp engine (if Council rejects changes)."""
        self.temp_engine = None
