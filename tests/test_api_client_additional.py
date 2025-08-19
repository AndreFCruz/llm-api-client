"""
Additional edge-case tests for APIClient not covered in the main test file.
"""
from unittest.mock import patch

from llm_api_client import APIClient


@patch("litellm.get_model_info")
def test_get_max_context_tokens_fallback_on_error(mock_get_model_info):
    mock_get_model_info.side_effect = RuntimeError("boom")
    client = APIClient()

    # When litellm raises, the client should fall back to default env value (100_000)
    max_tokens = client.get_max_context_tokens("any-model")
    assert isinstance(max_tokens, int)
    assert max_tokens >= 100_000


@patch("litellm.token_counter")
def test_count_messages_tokens_timeout_and_exception_paths(mock_token_counter):
    client = APIClient()
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]

    # Simulate timeout via concurrent.futures.TimeoutError from future.result
    class _Timeout(Exception):
        pass

    from concurrent.futures import TimeoutError as FuturesTimeout

    def timeout_side_effect(*args, **kwargs):
        raise FuturesTimeout()

    mock_token_counter.side_effect = timeout_side_effect
    approx_tokens = client.count_messages_tokens(messages, model="gpt-3.5-turbo", timeout=0.01)
    # Rough approximation is char_len // 3
    expected = (len("hello") + len("world")) // 3
    assert approx_tokens == expected

    # Generic exception path
    mock_token_counter.side_effect = RuntimeError("other error")
    approx_tokens2 = client.count_messages_tokens(messages, model="gpt-3.5-turbo")
    assert approx_tokens2 == expected


@patch("litellm.get_supported_openai_params")
def test_remove_unsupported_params_preserves_provider_specific(mock_get_params):
    mock_get_params.return_value = ["max_tokens", "temperature", "top_p"]

    client = APIClient()
    req = {
        "model": "some/provider-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "temperature": 0.5,
        "provider_x": 123,  # not in ALL_COMPLETION_PARAMS => should be preserved
        # Use a param that is OpenAI-known but not in supported list returned
        # by mock_get_params above, so it should be dropped.
        "tools": [],
    }

    cleaned = client.remove_unsupported_params(req)
    assert cleaned["model"] == req["model"]
    assert cleaned["messages"] == req["messages"]
    assert "max_tokens" in cleaned
    assert "provider_x" in cleaned  # preserved as provider-specific
    assert "tools" not in cleaned


@patch("litellm.get_supported_openai_params")
def test_remove_unsupported_params_filters_o_series_temperature(mock_get_params):
    mock_get_params.return_value = ["max_tokens", "temperature"]
    client = APIClient()
    req = {
        "model": "openai/o3-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10,
        "temperature": 0.7,  # should be removed for openai/o*
    }

    cleaned = client.remove_unsupported_params(req)
    assert "temperature" not in cleaned
    assert "max_tokens" in cleaned
