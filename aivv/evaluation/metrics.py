"""
Evaluation Metrics for ACA

Implements metrics for UUV anomaly detection:
- Recall (must maintain >0.95)
- Precision (goal to maximize)
- F1-Score (primary thesis metric)
- False Positive Rate (goal to minimize)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ACAMetrics:
    """
    Comprehensive metrics for ACA evaluation.
    """
    
    # Core counts
    true_positives: int = 0      # Correctly identified failures
    true_negatives: int = 0      # Correctly identified passes
    false_positives: int = 0     # False alarms (predicted fail, actually pass)
    false_negatives: int = 0     # Missed failures (predicted pass, actually fail)
    
    # ACA-specific counts
    overrides: int = 0           # Council overrode initial anomaly flag
    confirmations: int = 0       # Council confirmed initial anomaly flag
    
    # Running history
    predictions: List[int] = field(default_factory=list)
    actuals: List[int] = field(default_factory=list)
    council_decisions: List[str] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
    
    @property
    def recall(self) -> float:
        """
        Recall = TP / (TP + FN)
        
        Goal: Maintain >0.95 - Don't miss real failures!
        """
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator
    
    @property
    def precision(self) -> float:
        """
        Precision = TP / (TP + FP)
        
        Goal: Maximize this - This is where ACA beats standard LSTM.
        """
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator
    
    @property
    def f1_score(self) -> float:
        """
        F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        
        THE PRIMARY THESIS METRIC.
        """
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    @property
    def false_positive_rate(self) -> float:
        """
        FPR = FP / (FP + TN)
        
        Goal: Minimize this - Prove the Council reduces false alarms.
        """
        denominator = self.false_positives + self.true_negatives
        if denominator == 0:
            return 0.0
        return self.false_positives / denominator
    
    @property
    def accuracy(self) -> float:
        """Standard accuracy (not primary metric due to imbalance)."""
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total
    
    @property
    def specificity(self) -> float:
        """Specificity = TN / (TN + FP)"""
        denominator = self.true_negatives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_negatives / denominator
    
    @property
    def override_rate(self) -> float:
        """Rate at which Council overrides initial Sentry decisions."""
        total_council = self.overrides + self.confirmations
        if total_council == 0:
            return 0.0
        return self.overrides / total_council
    
    def update(
        self,
        prediction: int,
        actual: int,
        council_decision: Optional[str] = None
    ) -> None:
        """
        Update metrics with a new prediction.
        
        Args:
            prediction: Final predicted label (0=Pass, 1=Fail)
            actual: Actual label (0=Pass, 1=Fail)
            council_decision: Optional council decision ("OVERRIDE" or "CONFIRM")
        """
        self.predictions.append(prediction)
        self.actuals.append(actual)
        
        if prediction == 1 and actual == 1:
            self.true_positives += 1
        elif prediction == 0 and actual == 0:
            self.true_negatives += 1
        elif prediction == 1 and actual == 0:
            self.false_positives += 1
        else:  # prediction == 0 and actual == 1
            self.false_negatives += 1
        
        if council_decision:
            self.council_decisions.append(council_decision)
            if council_decision == "OVERRIDE":
                self.overrides += 1
            elif council_decision == "CONFIRM":
                self.confirmations += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            'total_samples': self.total,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'recall': self.recall,
            'precision': self.precision,
            'f1_score': self.f1_score,
            'false_positive_rate': self.false_positive_rate,
            'accuracy': self.accuracy,
            'specificity': self.specificity,
            'council_overrides': self.overrides,
            'council_confirmations': self.confirmations,
            'override_rate': self.override_rate
        }
    
    def print_report(self) -> str:
        """Generate printable report."""
        summary = self.get_summary()
        
        report = """
╔══════════════════════════════════════════════════════════════╗
║               ACA EVALUATION METRICS REPORT                  ║
╠══════════════════════════════════════════════════════════════╣
║ PRIMARY METRICS (UUV PROJECT GOALS)                          ║
╠──────────────────────────────────────────────────────────────╣
║  F1-Score:              {f1:.4f}  (Primary Metric)           ║
║  Recall:                {recall:.4f}  (Goal: >0.95)          ║
║  Precision:             {precision:.4f}  (Maximize)          ║
║  False Positive Rate:   {fpr:.4f}  (Minimize)                ║
╠──────────────────────────────────────────────────────────────╣
║ CONFUSION MATRIX                                             ║
╠──────────────────────────────────────────────────────────────╣
║                  Predicted                                   ║
║                 Pass    Fail                                 ║
║  Actual Pass    {tn:<5}   {fp:<5}   (TN / FP)                ║
║  Actual Fail    {fn:<5}   {tp:<5}   (FN / TP)                ║
╠──────────────────────────────────────────────────────────────╣
║ COUNCIL STATISTICS                                           ║
╠──────────────────────────────────────────────────────────────╣
║  Overrides:       {overrides:<5}                             ║
║  Confirmations:   {confirms:<5}                              ║
║  Override Rate:   {override_rate:.2%}                        ║
╚══════════════════════════════════════════════════════════════╝
""".format(
            f1=summary['f1_score'],
            recall=summary['recall'],
            precision=summary['precision'],
            fpr=summary['false_positive_rate'],
            tn=summary['true_negatives'],
            fp=summary['false_positives'],
            fn=summary['false_negatives'],
            tp=summary['true_positives'],
            overrides=summary['council_overrides'],
            confirms=summary['council_confirmations'],
            override_rate=summary['override_rate']
        )
        
        return report
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.overrides = 0
        self.confirmations = 0
        self.predictions = []
        self.actuals = []
        self.council_decisions = []


def compute_metrics(
    predictions: List[int],
    actuals: List[int]
) -> Dict[str, float]:
    """
    Compute metrics from prediction and actual lists.
    
    Args:
        predictions: List of predicted labels (0/1)
        actuals: List of actual labels (0/1)
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = ACAMetrics()
    
    for pred, actual in zip(predictions, actuals):
        metrics.update(pred, actual)
    
    return metrics.get_summary()


def compare_experiments(
    results: Dict[str, ACAMetrics]
) -> str:
    """
    Generate comparison report for ablation experiments.
    
    Args:
        results: Dict mapping experiment name to metrics
        
    Returns:
        Formatted comparison string
    """
    header = "| Experiment | F1-Score | Recall | Precision | FPR | Override Rate |"
    separator = "|------------|----------|--------|-----------|-----|---------------|"
    
    rows = []
    for name, metrics in results.items():
        row = f"| {name:<10} | {metrics.f1_score:.4f}   | {metrics.recall:.4f} | {metrics.precision:.4f}    | {metrics.false_positive_rate:.4f} | {metrics.override_rate:.2%}         |"
        rows.append(row)
    
    return "\n".join([header, separator] + rows)
