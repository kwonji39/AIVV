"""
GenAI LLM Client (Groq Commercial Version)

Client for Groq's OpenAI-compatible API.
Supports multi-model configuration, rate-limiting, native JSON mode, 
and robust regex parsing for reasoning models.
"""

import os
import json
import time
import random
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# --- LOGGING SETUP (warnings/errors only by default) ---
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("llm_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Default Groq API Configuration
GENAI_BASE_URL = "https://api.groq.com/openai/v1"
GENAI_DEFAULT_MODEL = "llama-3.3-70b-versatile" # You can change this in your config
GENAI_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GENAI_API_KEY", "")


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]


class GenAIClient:
    """
    Client for GenAI Inference Endpoints.
    Uses OpenAI-compatible API pointed at Groq.
    """
    
    def __init__(
        self,
        base_url: str = GENAI_BASE_URL,
        model: str = GENAI_DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or GENAI_API_KEY
        self._client = None
        
        self.last_request_time = 0.0
        self.min_delay = 0.5  # Lowered because Groq can handle high concurrency
    
    def _get_client(self):
        """Get or create OpenAI client connected to Groq."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Missing LLM API key. Set GROQ_API_KEY or GENAI_API_KEY in your environment."
                )
            try:
                from openai import OpenAI
                import httpx
                
                # Standard, fast timeout (Groq is usually instant)
                http_client = httpx.Client(timeout=60.0)
                
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    http_client=http_client,
                    max_retries=2 
                )
                logger.debug("Groq client initialized successfully.")
                
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai httpx")
        
        return self._client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        _retry_on_json_failure: bool = True,
        _force_json_mode: Optional[bool] = None,
    ) -> LLMResponse:
        """
        Send chat completion request with Native JSON mode.
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep((self.min_delay - elapsed) + random.uniform(0.1, 0.3))
            
        client = self._get_client()
        target_model = model or self.model
        
        try:
            logger.debug(f"Sending request to {target_model}...")
            start_time = time.time()
            
            # THE MAGIC BULLET: Check if the prompt asks for JSON
            wants_json = any("JSON" in str(m.get("content", "")).upper() for m in messages)
            use_json_mode = wants_json if _force_json_mode is None else _force_json_mode
            
            # Build request arguments
            kwargs = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            
            # Force Groq API to output valid JSON
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = client.chat.completions.create(**kwargs)
            
            self.last_request_time = time.time()
            duration = self.last_request_time - start_time
            logger.debug(f"Success: Received response from {target_model} in {duration:.2f}s")
            
            # Safely handle usage stats if missing
            usage_dict = {}
            if response.usage:
                usage_dict = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0)
                }
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=usage_dict,
                raw_response=response.model_dump()
            )
            
        except Exception as e:
            from openai import RateLimitError, APITimeoutError, APIConnectionError

            error_str = str(e)
            if _retry_on_json_failure and "json_validate_failed" in error_str:
                logger.warning("JSON validation failed at provider. Retrying once with stricter JSON instructions.")
                repaired_messages = self._build_json_retry_messages(messages)
                return self.chat(
                    repaired_messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    model=model,
                    _retry_on_json_failure=False,
                    _force_json_mode=False,
                )
            
            if isinstance(e, RateLimitError):
                logger.warning("Groq Rate Limit Hit (429). Sleeping 3s and retrying...")
                time.sleep(3)
                self.last_request_time = time.time() 
                return self.chat(messages, temperature, max_tokens, model, _retry_on_json_failure, _force_json_mode)
            else:
                logger.error(f"API Error ({target_model}): {str(e)}")
                raise e

    def _build_json_retry_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Append stronger formatting instructions for a single retry after provider JSON validation failure."""
        retry_suffix = (
            "\n\nJSON RETRY INSTRUCTIONS:\n"
            "- Return exactly one raw JSON object and nothing else.\n"
            "- All numeric fields must be written as digits, e.g. 0.83, not words like 'zero point eight three' or '0. eight'.\n"
            "- Do not include comments, markdown, trailing commas, NaN, Infinity, or explanatory text.\n"
            "- Ensure every key and string value uses double quotes.\n"
            "- If uncertain, choose a simple valid value such as confidence 0.70."
        )

        repaired = [dict(message) for message in messages]
        if repaired:
            repaired[-1] = {
                **repaired[-1],
                "content": f"{repaired[-1].get('content', '')}{retry_suffix}",
            }
        return repaired
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat(messages, **kwargs)
        return response.content
    
    def analyze_anomaly(
        self,
        context: Dict[str, Any],
        system_prompt: str
    ) -> Dict[str, Any]:
        """Analyze anomaly with bulletproof Regex JSON extraction."""
        prompt = (
            "Analyze this anomaly context and respond with one valid JSON object only.\n"
            f"Context:\n{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
        )
        
        logger.debug("Starting analyze_anomaly task...")
        response = self.complete(prompt, system_prompt=system_prompt)
        
        try:
            # 1. Strip out DeepSeek-style reasoning tags if they slip through
            clean_text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            
            # 2. Hunt for the exact JSON brackets
            match = re.search(r'\{.*\}', clean_text.strip(), re.DOTALL)
            
            if match:
                json_string = match.group(0)
                parsed_data = json.loads(json_string)
                logger.debug("Successfully parsed JSON output.")
                return parsed_data
            else:
                # If regex fails to find brackets, try parsing the raw string anyway
                return json.loads(clean_text)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response. Returning raw output.")
            return {"raw_response": response, "parse_error": True}
    

def get_llm_client(**kwargs) -> GenAIClient:
    return GenAIClient(**kwargs)
