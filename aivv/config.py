"""
AIVV Configuration Module

Contains all configuration settings including ablation study flags.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class ACAConfig:
    """Configuration for the AIVV anomaly detection pipeline."""

    # Domain mode for agent prompts/rules.
    # UUV-only project configuration.
    domain: str = "uuv"
    
    # ===== Reproducibility =====
    SEED: int = 42  # Random seed for LSTM and NumPy (set to None to disable)
    
    # ===== Ablation Study Flags =====
    # Experiment A: Disable Tuner (Agent 3)
    ENABLE_AGENT_3: bool = True
    
    # Experiment B: Disable Council (Group B)
    ENABLE_GROUP_B: bool = True

    # Temporal smoothing on Council reporting state only (online debouncing).
    # This no longer changes Inspector/Tuner control flow.
    ENABLE_COUNCIL_TEMPORAL_FILTER: bool = True
    council_confirm_k: int = 2   # require >=k consecutive FAILs to mark reporting failure state
    council_release_k: int = 2   # require >=k consecutive PASSes to clear reporting failure state
    
    # Experiment C: Full ACA (convenience flag)
    @property
    def ALL_AGENTS_ON(self) -> bool:
        return self.ENABLE_AGENT_3 and self.ENABLE_GROUP_B
    
    # ===== LLM Settings (Groq/OpenAI-compatible API) =====
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or os.getenv("GENAI_API_KEY", "")
    )
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    
    # Per-agent model assignments
    # Agent 2: Inspector - Strategy translator
    agent2_model: str = "qwen/qwen3-32b"
    # Agent 3: Tuner - Intelligent alpha recommendation
    agent3_model: str = "openai/gpt-oss-20b"
    # Agent 4: Requirements Engineer - Compliance check
    agent4_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    # Agent 5: Failure Manager - Risk assessment
    agent5_model: str = "openai/gpt-oss-120b"
    # Agent 6: System Engineer - LSTM-aware council voter
    agent6_model: str = "llama-3.3-70b-versatile"
    
    # # Agent 2: Inspector - Strategy translator
    # agent2_model: str = "openai/gpt-oss-20b"
    # # Agent 3: Tuner - Intelligent alpha recommendation
    # agent3_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    # # Agent 4: Requirements Engineer - Compliance check
    # agent4_model: str = "openai/gpt-oss-120b"
    # # Agent 5: Failure Manager - Risk assessment
    # agent5_model: str = "llama-3.3-70b-versatile"
    # # Agent 6: System Engineer - LSTM-aware council voter
    # agent6_model: str = "qwen/qwen3-32b"
    
    # ===== Engine Settings =====
    # LSTM Architecture
    lstm_input_dim: int = 1  # single yaw channel
    lstm_hidden_dim: int = 32
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # Variational/Bayesian settings
    mc_samples: int = 30  # Monte Carlo samples for uncertainty (30 before)
    
    # Conformal Prediction defaults
    default_alpha: float = 0.05  # 95% confidence
    recalibration_alpha: float = 0.01  # 99% confidence (for noise handling)

    # Agent-3-driven adaptive operating alpha (for Sentry + Engine)
    # If enabled, Council-approved recommended_alpha will be applied online.
    adaptive_alpha_enabled: bool = True
    adaptive_alpha_min: float = 0.01
    adaptive_alpha_max: float = 0.10
    adaptive_alpha_smoothing: float = 1.0  # 1.0 = immediate apply, <1.0 = EMA-style update

    # Sentry policy: if False, uncertainty alone does not trigger FAIL.
    # This helps suppress short-lived false alarms from transient spikes.
    sentry_uncertainty_only_fail: bool = False
    
    # Fine-tuning defaults
    fine_tune_lr: float = 0.0001
    fine_tune_epochs: int = 5
    
    # ===== Data Settings =====
    # Sliding window
    window_size: int = 20
    
    # ===== Paths =====
    data_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "data"))
    model_checkpoint_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "checkpoints"))
    logs_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "logs"))
    
    # ===== Device Settings =====
    # Preferred accelerator order when ACA_DEVICE is not explicitly set:
    # mps (Apple Silicon) -> cuda -> xpu -> cpu
    preferred_device: str = "auto"
    
    def get_device(self):
        """Get the appropriate torch device."""
        import torch
        requested = os.getenv("ACA_DEVICE", self.preferred_device).lower()

        # Explicit override from env/config
        if requested in {"cpu", "mps", "cuda", "xpu"}:
            if requested == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps")
            elif requested == "cuda":
                if torch.cuda.is_available():
                    return torch.device("cuda")
            elif requested == "xpu":
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    return torch.device("xpu")
            elif requested == "cpu":
                return torch.device("cpu")

            # If requested accelerator is unavailable, fall back to CPU.
            return torch.device("cpu")

        # Auto mode (best local default for Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    
