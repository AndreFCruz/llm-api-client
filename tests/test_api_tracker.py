"""
Additional tests for APIUsageTracker.

These tests exercise cost tracking, response time statistics, and integration
points with LiteLLM callbacks.
"""
from datetime import datetime, timedelta

import pytest

from llm_api_client.api_tracker import APIUsageTracker


class _TrackerMockResponse:
    """Minimal response object compatible with APIUsageTracker._log_response.

    Provides a `.usage` dict and supports `dict(response)` by implementing
    iteration over key-value pairs.
    """

    def __init__(self, *, prompt_tokens=10, completion_tokens=20, created=None):
        self.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        self.created = int((created or datetime.now()).timestamp())
        self.model = "test-model"
        self.choices = []

    def __iter__(self):
        # Yield a few representative fields so dict(self) works
        yield from (
            ("model", self.model),
            ("created", self.created),
            ("usage", self.usage),
        )


def test_initial_details_no_responses():
    """details should be sane with no responses logged."""
    tracker = APIUsageTracker()
    d = tracker.details
    assert d["total_cost"] == 0
    assert d["total_prompt_tokens"] == 0
    assert d["total_completion_tokens"] == 0
    assert d["num_api_calls"] == 0
    assert d["mean_response_time"] is None
    assert d["response_times"] == {}


def test_track_cost_callback_accumulates_and_logs(monkeypatch):
    """track_cost_callback should accumulate costs and response metadata."""
    tracker = APIUsageTracker()

    # Create two mock responses and timestamps one second apart
    resp1 = _TrackerMockResponse(prompt_tokens=5, completion_tokens=7)
    resp2 = _TrackerMockResponse(prompt_tokens=11, completion_tokens=13)

    start1 = datetime.now()
    end1 = start1 + timedelta(seconds=0.4)
    start2 = end1 + timedelta(seconds=0.2)
    end2 = start2 + timedelta(seconds=1.0)

    tracker.track_cost_callback(
        {"response_cost": 0.123}, resp1, start1, end1
    )
    tracker.track_cost_callback(
        {"response_cost": 0.877}, resp2, start2, end2
    )

    # Totals
    assert pytest.approx(tracker.total_cost, rel=1e-6) == 1.0
    assert tracker.total_prompt_tokens == 5 + 11
    assert tracker.total_completion_tokens == 7 + 13
    assert tracker.num_api_calls == 2

    # Response time stats
    assert tracker.mean_response_time is not None
    assert tracker.response_time_at_percentile(50) is not None

    details = tracker.details
    assert "50percentile" in details["response_times"] or "75percentile" in details["response_times"]


def test_get_stats_str_contains_key_metrics():
    tracker = APIUsageTracker()
    now = datetime.now()
    tracker.track_cost_callback({"response_cost": 0.5}, _TrackerMockResponse(), now, now + timedelta(seconds=0.2))
    s = tracker.get_stats_str()
    assert "Total cost of API calls: $" in s
    assert "Total prompt tokens:" in s
    assert "Total completion tokens:" in s
    assert "Number of responses:" in s


def test_set_up_litellm_cost_tracking_sets_callback(monkeypatch):
    import litellm

    tracker = APIUsageTracker()
    tracker.set_up_litellm_cost_tracking()

    assert isinstance(litellm.success_callback, list)
    assert tracker.track_cost_callback in litellm.success_callback
