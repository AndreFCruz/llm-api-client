"""Local LiteLLM stub for tests.

Provides minimal API used by the codebase and tests so we can run without the
real dependency and its transitive imports.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable


# Public variables expected by code
success_callback: list = []


def completion(*args, **kwargs):
    raise NotImplementedError("This stub should be patched in tests.")


def token_counter(*, model: str, messages: list[dict]) -> int:
    # Very rough token estimate: 1 token per 4 characters
    total_chars = sum(len(m.get("content", "")) for m in messages if m and m.get("content") is not None)
    return max(0, total_chars // 4)


def get_model_info(model: str) -> dict[str, Any]:
    # Return a reasonable default max input tokens
    default_max = 8192 if "gpt" in model.lower() else 4096
    return {"max_input_tokens": default_max}


def get_supported_openai_params(*, model: str, request_type: str) -> list[str]:
    # Minimal parameter set used in tests
    return [
        "temperature",
        "max_tokens",
        "messages",
        "model",
        "tools",
    ]


# Provide a logger named like the real package so code can adjust handlers
logging.getLogger("LiteLLM")

