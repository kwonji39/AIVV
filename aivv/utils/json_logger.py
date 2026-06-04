"""
ACA JSON Logger

Provides structured logging for agent decisions with timestamps.
Each run creates a unique log file with job ID and timestamp.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import asdict
import logging

from ..config import ACAConfig


class ACALogger:
    """
    JSON logger for ACA agent decisions.
    
    Creates structured logs with:
    - Unique job ID (timestamp-based or user-provided)
    - Per-sample decision trails
    - Agent reasoning chains
    """
    
    def __init__(
        self,
        config: Optional[ACAConfig] = None,
        job_id: Optional[str] = None,
        logs_dir: Optional[str] = None
    ):
        """
        Initialize the logger.
        
        Args:
            config: ACA configuration
            job_id: Optional job identifier (auto-generated if not provided)
            logs_dir: Directory for log files
        """
        self.config = config or ACAConfig()
        
        # Generate job ID based on timestamp if not provided
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.job_id = job_id or f"aca_run_{self.timestamp}"
        
        # Set up logs directory
        self.logs_dir = logs_dir or self.config.logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create log file path
        self.log_file = os.path.join(
            self.logs_dir, 
            f"{self.job_id}.json"
        )
        
        # Initialize log structure
        sample_logs: List[Dict[str, Any]] = []
        self.log_data = {
            "job_id": self.job_id,
            "timestamp": self.timestamp,
            "config": self._serialize_config(),
            "sample_logs": sample_logs,
            "summary": None
        }
        
        # Save initial log
        self._save_log()
        
        print(f"ACA Logger initialized: {self.log_file}")
    
    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize config to JSON-safe dict."""
        config_dict = {}
        for field in ['domain', 'llm_base_url', 'agent2_model', 'agent4_model', 
                      'agent5_model', 'agent6_model', 'llm_temperature',
                      'llm_max_tokens', 'ENABLE_AGENT_3', 'ENABLE_GROUP_B',
                      'default_alpha', 'window_size']:
            if hasattr(self.config, field):
                config_dict[field] = getattr(self.config, field)
        return config_dict
    
    def _save_log(self):
        """Save log data to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2, default=str)
    
    def log_sample_start(self, sample_id: int, actual_label: Optional[float]) -> Dict[str, Any]:
        """
        Start logging for a new sample.
        
        Returns sample log entry for subsequent agent logs.
        """
        actual_label_int = None if actual_label is None else int(actual_label)
        sample_log = {
            "sample_id": sample_id,
            "actual_label": actual_label_int,
            "actual_class": (
                "UNKNOWN"
                if actual_label_int is None
                else ("FAIL" if actual_label_int > 0 else "PASS")
            ),
            "start_time": datetime.now().isoformat(),
            "agent_decisions": [],
            "final_decision": None,
            "was_override": False,
            "correct": None
        }
        self.log_data["sample_logs"].append(sample_log)
        return sample_log
    
    def log_agent_decision(
        self,
        sample_id: int,
        agent_name: str,
        agent_type: str,
        model_used: str,
        decision: str,
        confidence: float,
        reasoning: str,
        payload: Optional[Dict[str, Any]] = None
    ):
        """
        Log an agent's decision for a sample.
        
        Args:
            sample_id: Sample identifier
            agent_name: Human-readable agent name
            agent_type: Agent type enum value
            model_used: LLM model used for this agent
            decision: PASS/FAIL/UNCERTAIN
            confidence: Confidence score
            reasoning: Agent's reasoning
            payload: Additional payload data
        """
        # Find the sample log
        sample_log = None
        for wl in self.log_data["sample_logs"]:
            if wl["sample_id"] == sample_id:
                sample_log = wl
                break
        
        if sample_log is None:
            logging.warning(f"No sample log found for sample_id={sample_id}")
            return
        
        # Create agent decision entry
        agent_entry = {
            "agent_name": agent_name,
            "agent_type": agent_type,
            "model_used": model_used,
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning[:500] if reasoning else None,  # Truncate long reasoning
            "timestamp": datetime.now().isoformat(),
            "payload_summary": self._summarize_payload(payload) if payload else None
        }
        
        sample_log["agent_decisions"].append(agent_entry)
        self._save_log()
    
    def _summarize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key info from payload for logging."""
        summary = {}
        
        # Key fields to extract
        key_fields = [
            'error', 'bound', 'is_anomaly', 'prediction',
            'vote_breakdown', 'weighted_score', 'action',
            'promote_temp_engine', 'had_veto', 'veto_reason',
            'compliance_status', 'risk_level'
        ]
        
        for field in key_fields:
            if field in payload:
                value = payload[field]
                # Handle tensor values
                if hasattr(value, 'item'):
                    value = value.item()
                elif hasattr(value, 'tolist'):
                    value = value.tolist()
                summary[field] = value
        
        return summary
    
    def log_sample_complete(
        self,
        sample_id: int,
        final_decision: str,
        was_override: bool,
        correct: bool
    ):
        """
        Complete logging for a sample.
        
        Args:
            sample_id: Sample identifier
            final_decision: Final PASS/FAIL decision
            was_override: Whether Council overrode initial decision
            correct: Whether final decision matched actual label
        """
        for sample_log in self.log_data["sample_logs"]:
            if sample_log["sample_id"] == sample_id:
                sample_log["final_decision"] = final_decision
                sample_log["was_override"] = was_override
                sample_log["correct"] = correct
                sample_log["end_time"] = datetime.now().isoformat()
                break
        
        self._save_log()
    
    def log_summary(
        self,
        metrics: Dict[str, Any],
        council_stats: Dict[str, Any]
    ):
        """
        Log final summary statistics.
        
        Args:
            metrics: Evaluation metrics (F1, recall, precision, etc.)
            council_stats: Council override statistics
        """
        self.log_data["summary"] = {
            "metrics": metrics,
            "council_stats": council_stats,
            "total_samples": len(self.log_data["sample_logs"]),
            "end_time": datetime.now().isoformat()
        }
        self._save_log()
        
        print(f"ACA Log saved: {self.log_file}")
    
    def get_sample_log(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """Get log entry for a specific sample."""
        for sample_log in self.log_data["sample_logs"]:
            if sample_log["sample_id"] == sample_id:
                return sample_log
        return None
    
    def get_all_logs(self) -> Dict[str, Any]:
        """Get complete log data."""
        return self.log_data


def load_aca_log(log_file: str) -> Dict[str, Any]:
    """Load an ACA log file."""
    with open(log_file, 'r') as f:
        return json.load(f)


def list_aca_logs(logs_dir: str) -> List[str]:
    """List all ACA log files in a directory."""
    if not os.path.exists(logs_dir):
        return []
    
    return sorted([
        f for f in os.listdir(logs_dir) 
        if f.startswith('aca_run_') and f.endswith('.json')
    ], reverse=True)  # Most recent first
