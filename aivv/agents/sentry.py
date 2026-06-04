"""
Agent 1: The Sentry (Binary Gatekeeper)

Strict enforcement agent using conformal prediction checks.
NON-ADAPTIVE - applies fixed rules without learning.
"""

from typing import Optional, Tuple
import numpy as np
from .base import BaseAgent, AgentResult, AgentType, Decision, CaseFile
from ..engine.engine import UUVEngine
from ..config import ACAConfig


class SentryAgent(BaseAgent):
    """
    The Sentry: Binary Gatekeeper
    
    Role: Strict enforcement using conformal prediction
    Method: Conformal Prediction Check
    Non-Adaptive: Uses fixed alpha (95% confidence)
    
    Process:
        1. Call Engine.predict(current_wafer)
        2. Calculate Standard Interval using α=0.05
        3. IF |Error| > Interval: FAIL
           IF |Error| <= Interval: PASS
    
    Trigger: FAIL automatically triggers the Council
    """
    
    def __init__(
        self,
        engine: UUVEngine,
        config: Optional[ACAConfig] = None
    ):
        super().__init__(AgentType.SENTRY, "Sentry")
        self.engine = engine
        self.config = config or ACAConfig()
        self.alpha = self.config.default_alpha  # 0.05 for 95% confidence
    
    def execute(
        self,
        case_file: CaseFile,
        window=None,
        actual: Optional[float] = None,
        **kwargs
    ) -> AgentResult:
        """
        Execute anomaly detection check.
        
        Uses classification-based detection:
        - If model predicts high failure probability (>threshold), flag as potential anomaly
        - Combined with uncertainty: high uncertainty = more likely to flag
        
        Args:
            case_file: Current case file
            window: Input window for prediction (optional if already in case_file)
            actual: Actual value (optional if already in case_file)
            
        Returns:
            AgentResult with PASS or FAIL decision
        """
        # Get prediction if window provided
        if window is not None:
            prediction, uncertainty = self.engine.predict(window)
            case_file.prediction = prediction
            case_file.uncertainty = uncertainty
        
        # Use provided actual or from case file
        if actual is not None:
            case_file.actual = actual
        
        # Primary detection mechanism is CONFORMAL PREDICTION:
        # If |prediction - actual| exceeds the conformal interval,
        # the sample is non-conforming → FAIL.
        #
        # Secondary signals (prediction magnitude + uncertainty) provide
        # early warning but should NOT override the conformal check.
        
        prediction = case_file.prediction
        uncertainty = case_file.uncertainty if case_file.uncertainty else 0.0
        
        # Get conformal interval
        interval_width = self.engine.get_conformal_interval(self.alpha)
        case_file.bound = interval_width
        case_file.alpha = self.alpha
        
        # Compute prediction error against actual
        error = abs(prediction - case_file.actual)
        case_file.error = error
        
        # PRIMARY CHECK: Conformal prediction conformity test
        # This is the mathematically-grounded detection mechanism.
        exceeds_bound = error > interval_width
        
        # SECONDARY CHECK: High uncertainty means the model is confused.
        # Use calibrated uncertainty threshold when available.
        uncertainty_threshold = self.engine.get_uncertainty_threshold(self.alpha)
        if not np.isfinite(uncertainty_threshold):
            uncertainty_threshold = 0.2  # Fallback when uncertainty is not calibrated
        high_uncertainty = uncertainty > uncertainty_threshold
        
        # Final decision:
        # - Always FAIL if conformal bound is exceeded.
        # - Optionally FAIL on uncertainty alone (configurable).
        uncertainty_only_fail = bool(getattr(self.config, "sentry_uncertainty_only_fail", False))
        trigger_fail = exceeds_bound or (high_uncertainty and uncertainty_only_fail)
        is_pass = not trigger_fail
        decision = Decision.PASS if is_pass else Decision.FAIL
        
        # Anomaly score for confidence calculation
        anomaly_score = error / max(interval_width, 1e-8)  # Ratio of error to bound
        
        # Compute confidence
        if is_pass:
            confidence = max(0.5, 0.95 - 0.3 * anomaly_score)
        else:
            confidence = min(0.99, 0.6 + 0.15 * anomaly_score)
        
        result = AgentResult(
            agent_type=self.agent_type,
            decision=decision,
            confidence=max(0.5, min(1.0, confidence)),
            reasoning=f"Prediction={prediction:.4f}, Error={error:.4f}, "
                     f"Bound={interval_width:.4f} (α={self.alpha}), "
                     f"Uncertainty={uncertainty:.4f} (bound={uncertainty_threshold:.4f}), "
                     f"uncertainty_only_fail={uncertainty_only_fail}. "
                     f"{'Within conformal/policy bounds' if is_pass else 'Policy triggered FAIL - triggering Council'}.",
            payload={
                'prediction': prediction,
                'actual': case_file.actual,
                'error': error,
                'bound': interval_width,
                'alpha': self.alpha,
                'uncertainty': uncertainty,
                'uncertainty_bound': uncertainty_threshold,
                'anomaly_score': anomaly_score,
                'exceeds_bound': exceeds_bound,
                'high_uncertainty': high_uncertainty,
                'uncertainty_only_fail': uncertainty_only_fail,
                'trigger_fail': trigger_fail,
            }
        )
        
        # Add to case file
        case_file.add_result(result)
        
        # Log execution
        self.log_execution(case_file, result)
        
        return result
    
    def reevaluate(
        self,
        case_file: CaseFile,
        new_prediction: float,
        new_bound: float,
        new_uncertainty: Optional[float] = None,
        new_alpha: Optional[float] = None,
    ) -> AgentResult:
        """
        Re-evaluate with new prediction/bound from Tuner.
        
        Used by the Tuner's reevaluation candidate to check if the adjustment worked.
        
        Args:
            case_file: Case file to reevaluate
            new_prediction: New prediction from temp engine
            new_bound: New interval width
            
        Returns:
            AgentResult with new decision
        """
        original_prediction = case_file.prediction
        original_error = case_file.error
        original_bound = case_file.bound

        if new_uncertainty is not None:
            case_file.uncertainty = float(new_uncertainty)
        if new_alpha is not None:
            case_file.alpha = float(new_alpha)

        # Calculate new error
        error = abs(new_prediction - case_file.actual)

        # Refresh case file so any later council loop reasons on updated state.
        case_file.prediction = float(new_prediction)
        case_file.bound = float(new_bound)
        case_file.error = float(error)
        
        # Make new decision
        is_pass = error <= new_bound
        decision = Decision.PASS if is_pass else Decision.FAIL
        
        result = AgentResult(
            agent_type=self.agent_type,
            decision=decision,
            confidence=0.9 if is_pass else 0.6,
            reasoning=f"Re-evaluation: Error={error:.4f}, NewBound={new_bound:.4f}. "
                     f"{'Adjustment successful' if is_pass else 'Adjustment insufficient'}.",
            payload={
                'original_prediction': original_prediction,
                'new_prediction': new_prediction,
                'actual': case_file.actual,
                'original_error': original_error,
                'new_error': error,
                'original_bound': original_bound,
                'new_bound': new_bound,
                'original_alpha': self.alpha,
                'new_alpha': case_file.alpha,
                'reevaluation': True
            }
        )
        
        return result
    
