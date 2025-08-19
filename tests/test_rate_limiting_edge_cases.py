"""
Additional rate-limiting edge case tests for `APIClient`.

These focus on RPM and TPM interactions, zero-token requests, and early-fail
behavior via `max_delay_seconds` to avoid long test runtimes.
"""

import time
from unittest.mock import patch

import pytest

from llm_api_client import APIClient
from mock_objects import MockResponse


@patch("litellm.completion")
@patch("litellm.token_counter")
def test_tpm_max_delay_seconds_stops_blocking_early(mock_token_counter, mock_completion):
    """If TPM would block, small max_delay should cause early failure and skip API call."""
    # First request uses 1 token (OK), second uses 2 tokens which exceeds TPM=1 and should fail fast
    mock_token_counter.side_effect = [1, 2]
    mock_completion.return_value = MockResponse("ok")

    client = APIClient(
        max_requests_per_minute=None,
        max_tokens_per_minute=1,
        max_workers=2,
        max_delay_seconds=1,
    )

    requests = [
        {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": f"Test {i}"}]} for i in range(2)
    ]

    start = time.time()
    results = client.make_requests(requests, max_workers=2, sanitize=False)
    elapsed = time.time() - start

    # Exactly one success (TPM acquired) and one early failure (None)
    assert sum(r is not None for r in results) == 1
    assert sum(r is None for r in results) == 1
    assert mock_completion.call_count == 1
    assert elapsed <= 3, f"Expected early return due to max_delay; took {elapsed:.2f}s"


@patch("litellm.completion")
@patch("litellm.token_counter")
def test_zero_token_request_skips_tpm_weight(mock_token_counter, mock_completion):
    """Zero-token requests should not attempt to acquire TPM tokens; should proceed quickly."""
    mock_token_counter.return_value = 0
    mock_completion.return_value = MockResponse("ok")

    client = APIClient(
        max_requests_per_minute=None,
        max_tokens_per_minute=1,
        max_workers=1,
        max_delay_seconds=1,
    )

    request = {"model": "gpt-3.5-turbo", "messages": []}
    start = time.time()
    results = client.make_requests([request], sanitize=False)
    elapsed = time.time() - start

    assert len(results) == 1 and results[0] is not None
    assert mock_completion.call_count == 1
    assert elapsed <= 1.0


@patch("litellm.completion")
@patch("litellm.token_counter")
def test_both_limiters_active_tpm_failure_early(mock_token_counter, mock_completion):
    """With both RPM and TPM enabled, if TPM weight cannot be acquired within max_delay, skip API call."""
    mock_token_counter.return_value = 2
    mock_completion.return_value = MockResponse("ok")

    client = APIClient(
        max_requests_per_minute=1000,  # RPM not the bottleneck
        max_tokens_per_minute=1,       # TPM is the bottleneck
        max_workers=1,
        max_delay_seconds=1,
    )

    request = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}]}
    results = client.make_requests([request], sanitize=False)

    # Should fail due to TPM and not call completion
    assert results == [None]
    assert mock_completion.call_count == 0


@patch("litellm.completion")
def test_no_limits_does_not_block(mock_completion):
    """When both limits are None, requests should proceed without rate limiter involvement."""
    mock_completion.return_value = MockResponse("ok")

    client = APIClient(
        max_requests_per_minute=None,
        max_tokens_per_minute=None,
        max_workers=5,
    )

    requests = [
        {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": f"req {i}"}]}
        for i in range(5)
    ]

    results = client.make_requests(requests, sanitize=False)
    assert len(results) == 5
    assert mock_completion.call_count == 5


@patch("litellm.completion")
def test_rpm_zero_blocks_all_requests(mock_completion):
    """RPM=0 should make acquisition impossible; all requests return None, and no API call occurs."""
    mock_completion.return_value = MockResponse("ok")

    client = APIClient(
        max_requests_per_minute=0,
        max_tokens_per_minute=None,
        max_workers=2,
        max_delay_seconds=1,
    )

    requests = [
        {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": f"req {i}"}]}
        for i in range(2)
    ]

    results = client.make_requests(requests, sanitize=False)
    assert results == [None, None]
    assert mock_completion.call_count == 0


@patch("litellm.completion")
@patch("litellm.token_counter")
def test_tpm_zero_blocks_all_requests(mock_token_counter, mock_completion):
    """TPM=0 should make token acquisition impossible; all requests return None, and no API call occurs."""
    mock_token_counter.return_value = 1
    mock_completion.return_value = MockResponse("ok")

    client = APIClient(
        max_requests_per_minute=None,
        max_tokens_per_minute=0,
        max_workers=2,
        max_delay_seconds=1,
    )

    requests = [
        {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": f"req {i}"}]}
        for i in range(2)
    ]

    results = client.make_requests(requests, sanitize=False)
    assert results == [None, None]
    assert mock_completion.call_count == 0


@patch("litellm.completion")
@patch("litellm.token_counter")
def test_sanitize_false_still_enforces_tpm(mock_token_counter, mock_completion):
    """Even with sanitize=False, token counting and TPM enforcement should occur."""
    mock_token_counter.return_value = 5
    mock_completion.return_value = MockResponse("ok")

    client = APIClient(
        max_requests_per_minute=None,
        max_tokens_per_minute=1,
        max_workers=1,
        max_delay_seconds=1,
    )

    request = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 2.0,  # ignored by sanitize=False test, but shouldn't affect TPM
    }

    results = client.make_requests([request], sanitize=False)
    assert results == [None]
    assert mock_completion.call_count == 0


@patch("litellm.completion")
@patch("litellm.token_counter")
def test_rate_limit_failures_are_recorded_in_history(mock_token_counter, mock_completion):
    """When rate limit acquisition fails, history should still record the request with None response/content."""
    mock_token_counter.return_value = 5
    mock_completion.return_value = MockResponse("ok")

    client = APIClient(
        max_requests_per_minute=None,
        max_tokens_per_minute=1,
        max_workers=1,
        max_delay_seconds=1,
    )

    reqs = [
        {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "A"}]},
        {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "B"}]},
    ]

    results = client.make_requests(reqs, sanitize=False)
    # Both should fail due to TPM=1 and token_count=5
    assert results == [None, None]
    assert len(client.history) == 2
    assert client.history[0]["response"] is None
    assert client.history[0]["content"] is None
    assert client.history[1]["response"] is None
    assert client.history[1]["content"] is None

