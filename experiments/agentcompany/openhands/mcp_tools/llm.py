"""
LLM utilities for the synthetic pipeline.

Uses OpenAI client with optional LiteLLM proxy for multi-provider routing.
"""

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

_client = None
_using_litellm = False


def get_openai_client() -> OpenAI:
    """Get an OpenAI client configured from LLM_API_KEY / LLM_BASE_URL env vars."""
    global _client, _using_litellm
    if _client is not None:
        return _client

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "LLM_API_KEY environment variable is required. "
            "Set with: export LLM_API_KEY='your-key' or create a .env file"
        )

    base_url = os.environ.get("LLM_BASE_URL") or None
    _using_litellm = base_url is not None
    if not _using_litellm:
        logger.warning("LLM_BASE_URL not set, using default OpenAI endpoint")

    _client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=0,
    )
    return _client


def _strip_provider_prefix(model: str) -> str:
    """Strip LiteLLM routing prefixes (e.g. 'openai/gpt-4.1') for direct API calls."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def llm_completion(model: str, messages: list, **kwargs) -> str:
    """Single LLM completion attempt (no retries -- caller handles retry logic).

    This is intentionally retry-free because mcp_server.py wraps calls with
    its own @retry decorator budgeted against the 30s MCP timeout.
    """
    client = get_openai_client()
    effective_model = model if _using_litellm else _strip_provider_prefix(model)
    response = client.chat.completions.create(model=effective_model, messages=messages, **kwargs)
    if not response.choices or not response.choices[0].message:
        raise ValueError(f"Empty LLM response from {model}")
    return response.choices[0].message.content or ""
