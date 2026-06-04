"""
Base Agent Classes

Defines the common interface for all ACA agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from enum import Enum
import json
from datetime import datetime
import logging


class AgentType(Enum):
    """Types of agents in ACA."""
    SENTRY = "sentry"                    # Agent 1
    INSPECTOR = "inspector"              # Agent 2
    TUNER = "tuner"                      # Agent 3
    REQUIREMENTS_ENGINEER = "req_eng"    # Agent 4
    FAILURE_MANAGER = "fail_mgr"         # Agent 5
    SYSTEM_ENGINEER = "sys_eng"          # Agent 6


class Decision(Enum):
    """Agent decision types."""
    PASS = "PASS"
    FAIL = "FAIL"
    RECALIBRATE = "RECALIBRATE"
    FINE_TUNE = "FINE_TUNE"
    TRY_BOTH = "TRY_BOTH"
    UNKNOWN = "UNKNOWN"


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_type: AgentType
    decision: Decision
    confidence: float = 1.0
    reasoning: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent': self.agent_type.value,
            'decision': self.decision.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'payload': self.payload,
            'timestamp': self.timestamp
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class CaseFile:
    """
    Case file passed between agents.
    Contains all relevant information about a potential anomaly.
    """
    sample_id: int
    prediction: float = 0.0
    actual: float = 0.0
    error: float = 0.0
    bound: float = 0.0
    alpha: float = 0.05
    uncertainty: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    agent_results: List[AgentResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_id': self.sample_id,
            'prediction': self.prediction,
            'actual': self.actual,
            'error': self.error,
            'bound': self.bound,
            'alpha': self.alpha,
            'uncertainty': self.uncertainty,
            'history': self.history,
            'agent_results': [r.to_dict() for r in self.agent_results],
        }
    
    def add_result(self, result: AgentResult) -> None:
        self.agent_results.append(result)
    
    def get_latest_result(self) -> Optional[AgentResult]:
        return self.agent_results[-1] if self.agent_results else None


class BaseAgent(ABC):
    """
    Base class for all ACA agents.
    
    Defines the common interface and logging capabilities.
    """
    
    def __init__(self, agent_type: AgentType, name: str):
        self.agent_type = agent_type
        self.name = name
        self.execution_log: List[Dict[str, Any]] = []
    
    @abstractmethod
    def execute(self, case_file: CaseFile, **kwargs) -> AgentResult:
        """
        Execute the agent's core logic.
        
        Args:
            case_file: The case file to process
            **kwargs: Additional arguments
            
        Returns:
            AgentResult with decision and metadata
        """
        pass
    
    def log_execution(
        self,
        case_file: CaseFile,
        result: AgentResult,
        additional_info: Optional[Dict] = None
    ) -> None:
        """Log an execution for auditing."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.name,
            'sample_id': case_file.sample_id,
            'decision': result.decision.value,
            'confidence': result.confidence,
            'reasoning': result.reasoning
        }
        if additional_info:
            log_entry.update(additional_info)
        
        self.execution_log.append(log_entry)
    
    def get_log(self) -> List[Dict[str, Any]]:
        """Get execution log."""
        return self.execution_log
    
    def __repr__(self) -> str:
        return f"{self.name}({self.agent_type.value})"


class LLMAgent(BaseAgent):
    """
    Base class for LLM-powered agents.
    
    Provides common LLM interaction utilities.
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        name: str,
        llm_client,
        system_prompt: str
    ):
        super().__init__(agent_type, name)
        self.llm_client = llm_client
        self.system_prompt = system_prompt
    
    def query_llm(
        self,
        context: Dict[str, Any],
        additional_prompt: str = ""
    ) -> Dict[str, Any]:
        """
        Query the LLM with structured output.
        
        Args:
            context: Context information
            additional_prompt: Additional instructions
            
        Returns:
            Parsed JSON response from LLM
        """
        full_prompt = self.system_prompt
        if additional_prompt:
            full_prompt = f"{full_prompt}\n\n{additional_prompt}"
        
        try:
            return self.llm_client.analyze_anomaly(context, full_prompt)
        except Exception as e:
            logging.warning(
                f"[{self.name}] LLM request failed, using fallback defaults: {type(e).__name__}: {e}"
            )
            return {
                "fallback": True,
                "error_type": type(e).__name__,
                "error": str(e),
            }
    
    def parse_decision(self, llm_response: Dict[str, Any]) -> Decision:
        """Parse LLM response into a Decision enum."""
        # Handle voting responses
        vote = llm_response.get('vote', '').upper()
        if vote in ['PASS', 'APPROVED', 'OK', 'YES']:
            return Decision.PASS
        if vote in ['FAIL', 'REJECTED', 'NO', 'VETO']:
            return Decision.FAIL
        
        # Handle action responses
        action = llm_response.get('action', '').upper()
        suspicion = llm_response.get('suspicion', '').upper()
        
        if 'RECALIB' in action or 'NOISE' in suspicion:
            return Decision.RECALIBRATE
        if 'FINE' in action or 'TUNE' in action or 'DRIFT' in suspicion:
            return Decision.FINE_TUNE
        if 'BOTH' in action:
            return Decision.TRY_BOTH
        
        return Decision.UNKNOWN
