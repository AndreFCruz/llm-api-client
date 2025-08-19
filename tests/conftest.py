"""Fixtures for pytest tests in the project.
"""
import sys
import types
import pytest


# Provide lightweight import-time stubs for litellm and openai so tests can run
# without importing heavy external libraries with version constraints.
if "litellm" not in sys.modules:
    litellm = types.ModuleType("litellm")
    # attributes used in code/tests
    litellm.success_callback = []

    def _token_counter(**kwargs):
        # Default simplistic behavior; tests patch this as needed
        messages = kwargs.get("messages", []) or []
        return sum(len(m.get("content", "")) for m in messages) // 3

    def _get_supported_openai_params(**kwargs):
        return ["max_tokens", "temperature", "top_p", "tools"]

    def _get_model_info(model):
        # Use a large default context window so truncation converges near limit
        # in tests relying on heuristic behavior.
        return {"max_input_tokens": 20000}

    def _completion(**kwargs):
        raise RuntimeError("This stub should be patched in tests.")

    litellm.token_counter = _token_counter
    litellm.get_supported_openai_params = _get_supported_openai_params
    litellm.get_model_info = _get_model_info
    litellm.completion = _completion
    sys.modules["litellm"] = litellm

if "openai" not in sys.modules:
    # Minimal stub so that `openai.APIError` can be referenced in exception
    # handling paths. We don't need functionality.
    openai = types.ModuleType("openai")

    class _APIError(Exception):
        def __init__(self, message="", status_code=None, llm_provider=None):
            super().__init__(message)
            self.message = message
            self.status_code = status_code
            self.llm_provider = llm_provider

    openai.APIError = _APIError
    sys.modules["openai"] = openai


@pytest.fixture(params=[
    # Test case 1: Long message at the end
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "a tree " * 10_000}  # Very long message
    ],
    # Test case 2: Long message in the middle
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "a tree " * 10_000},  # Very long message
        {"role": "assistant", "content": "I'll help you with trees."}
    ],
    # Test case 3: Multiple messages of similar length
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "A " * 5000},
        {"role": "assistant", "content": "B " * 5000},
        {"role": "user", "content": "C " * 5000}
    ],
    # Test case 4: Mixed message lengths with long system prompt
    [
        {"role": "system", "content": "You are a helpful assistant. " * 5000},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Tell me about trees. " * 500}
    ],
    # Test case 5: Short content but many messages
    [{"role": ("user" if i % 2 == 0 else "assistant"), "content": f"Message {i}" * 100}
     for i in range(20)]
], ids=["long_end", "long_middle", "multiple_similar", "long_system", "many_messages"])
def test_messages(request):
    """Fixture providing various message configurations for testing.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest request object

    Returns
    -------
    list
        A list of message dictionaries with different configurations
    """
    return request.param
