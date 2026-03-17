"""
LLM utilities for the synthetic pipeline.

Uses OpenAI client with optional LiteLLM proxy for multi-provider routing.
"""

import logging
import os
from typing import Any, Dict

from openai import OpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# =============================================================================
# JSON SCHEMAS FOR STRUCTURED OUTPUTS
# =============================================================================

# Schema for segment extraction (matches Segment dataclass fields)
# NOTE: OpenAI strict mode requires ALL properties in `required` - nullable fields use type: ["X", "null"]
SEGMENT_EXTRACTION_SCHEMA: Dict[str, Any] = {
    "name": "segment_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Exact verbatim text span from prompt (with context)",
                        },
                        "value": {
                            "type": "string",
                            "description": "Shortest exact substring from prompt containing key info",
                        },
                        "dimension": {
                            "type": "string",
                            "enum": ["goal", "constraint", "input", "context"],
                        },
                        "subdimension": {
                            "type": "string",
                            "description": "Specific category from taxonomy",
                        },
                        "is_used_in_trajectory": {
                            "type": ["boolean", "null"],
                            "description": "Was value used in trajectory? null if no trajectory",
                        },
                        "first_use_pct": {
                            "type": ["number", "null"],
                            "description": "When first used (0.0-1.0). null if not used",
                        },
                        "checkpoint_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Affected checkpoints",
                        },
                        "criticality": {
                            "type": "number",
                            "description": "0.0-1.0: task fails without this?",
                        },
                        "guessability": {
                            "type": "number",
                            "description": "0.0-1.0: agent can infer?",
                        },
                    },
                    "required": [
                        "text",
                        "value",
                        "dimension",
                        "subdimension",
                        "is_used_in_trajectory",
                        "first_use_pct",
                        "checkpoint_refs",
                        "criticality",
                        "guessability",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["segments"],
        "additionalProperties": False,
    },
}

# Schema for variant generation (matches expected output fields)
VARIANT_GENERATION_SCHEMA: Dict[str, Any] = {
    "name": "variant_generation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "underspecified_prompt": {
                "type": "string",
                "description": "Modified prompt with segment removed. MUST preserve original formatting exactly.",
            },
            "expected_questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_id": {
                            "type": "string",
                            "description": "Segment ID (1-indexed), e.g. S1, S2",
                        },
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Clarifying questions for this segment",
                        },
                    },
                    "required": ["segment_id", "questions"],
                    "additionalProperties": False,
                },
                "description": "List of segment IDs with their respective clarifying questions",
            },
            "failure_mode": {
                "type": "string",
                "description": "What goes wrong if agent assumes wrong value",
            },
        },
        "required": ["underspecified_prompt", "expected_questions", "failure_mode"],
        "additionalProperties": False,
    },
}
logging.getLogger("httpx").setLevel(logging.WARNING)

# Module-level client (initialized lazily)
_client = None


def get_openai_client() -> OpenAI:
    """
    Get an OpenAI client, optionally configured for LiteLLM proxy.

    Environment variables:
        LLM_API_KEY: Required. Your API key
        LLM_BASE_URL: Optional. Proxy URL (omit to call providers directly)
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "LLM_API_KEY environment variable is required. Set in .env and run: source .env"
        )

    base_url = os.environ.get("LLM_BASE_URL") or None

    _client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,  # Retries handled by tenacity for better logging
    )
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=60, exp_base=1),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def llm_completion(model: str, messages: list, **kwargs) -> str:
    """
    LLM completion with automatic retries.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o" for LiteLLM proxy)
        messages: List of message dicts with "role" and "content"
        **kwargs: Additional arguments passed to chat.completions.create

    Returns:
        Response text from the model
    """
    client = get_openai_client()

    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    if not response.choices or not response.choices[0].message:
        raise ValueError(f"Empty LLM response from {model}")
    return response.choices[0].message.content or ""
