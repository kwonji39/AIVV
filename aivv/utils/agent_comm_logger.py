"""
Agent Communication Logger

Logs all inter-agent communications as JSON files in run-specific directories.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional


class AgentCommLogger:
    """
    Logs agent-to-agent communications as JSON files.
    
    Structure:
        logs/run_<timestamp>/
            sample_<id>/
                council_to_inspector.json
                inspector_to_tuner.json
                tuner_to_sentry.json
                council_decision.json
            run_summary.json
    """
    
    def __init__(self, base_dir: str = "logs", run_dir: Optional[str] = None):
        """Initialize logger with a run directory (new timestamped directory by default)."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_dir or os.path.join(base_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.current_sample_id: Optional[int] = None
        self.current_sample_dir: Optional[str] = None
        
        print(f"Agent Communication Logger: {self.run_dir}")
    
    def set_sample(self, sample_id: int) -> None:
        """Set current sample and create its directory."""
        self.current_sample_id = sample_id
        self.current_sample_dir = os.path.join(self.run_dir, f"sample_{sample_id}")
        os.makedirs(self.current_sample_dir, exist_ok=True)
    
    def log_communication(
        self,
        from_agent: str,
        to_agent: str,
        payload: Dict[str, Any],
        comm_type: Optional[str] = None
    ) -> str:
        """
        Log a communication between two agents.
        
        Args:
            from_agent: Source agent name (e.g., 'council', 'inspector')
            to_agent: Destination agent name
            payload: The data being communicated
            comm_type: Optional type override for filename
            
        Returns:
            Path to the created JSON file
        """
        if self.current_sample_dir is None:
            raise ValueError("Must call set_sample() before logging")
        
        # Create filename
        if comm_type:
            filename = f"{comm_type}.json"
        else:
            filename = f"{from_agent}_to_{to_agent}.json"
        
        filepath = os.path.join(self.current_sample_dir, filename)
        
        # Build log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sample_id": self.current_sample_id,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "payload": payload
        }
        
        # Write JSON file
        with open(filepath, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
        
        return filepath
    
    def read_communication(
        self,
        from_agent: str,
        to_agent: str,
        comm_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Read a previous communication.
        
        Args:
            from_agent: Source agent name
            to_agent: Destination agent name
            comm_type: Optional type override for filename
            
        Returns:
            The payload from the communication, or None if not found
        """
        if self.current_sample_dir is None:
            return None
        
        if comm_type:
            filename = f"{comm_type}.json"
        else:
            filename = f"{from_agent}_to_{to_agent}.json"
        
        filepath = os.path.join(self.current_sample_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data.get("payload")
    
    def log_final_decision(self, decision: Dict[str, Any]) -> str:
        """Log the final council decision for this sample."""
        return self.log_communication(
            from_agent="council",
            to_agent="final",
            payload=decision,
            comm_type="council_decision"
        )

    def save_run_artifact(self, name: str, payload: Dict[str, Any]) -> str:
        """Save a run-level JSON artifact inside the run directory."""
        filepath = os.path.join(self.run_dir, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        return filepath

    def save_sample_artifact(self, name: str, payload: Dict[str, Any]) -> str:
        """Save a sample-level JSON artifact inside the current sample directory."""
        if self.current_sample_dir is None:
            raise ValueError("Must call set_sample() before saving sample artifacts")

        filepath = os.path.join(self.current_sample_dir, f"{name}.json")
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        return filepath
    
    def save_run_summary(self, summary: Dict[str, Any]) -> str:
        """Save overall run summary."""
        filepath = os.path.join(self.run_dir, "run_summary.json")
        
        summary["run_timestamp"] = self.timestamp
        summary["saved_at"] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return filepath
    
    def get_run_dir(self) -> str:
        """Get the current run directory path."""
        return self.run_dir
