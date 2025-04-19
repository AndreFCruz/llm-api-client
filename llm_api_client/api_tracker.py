"""API usage tracker.
"""
import logging
from typing import Any

import numpy as np


class APIUsageTracker:
    """Class to track the cost of API calls."""

    def __init__(self):
        """Initialize the API usage tracker."""
        self._total_cost = 0
        self._responses = []

    def _log_response(self, response, start_time, end_time):
        """Log the query for debugging purposes."""
        response_dict = {
            "prompt_tokens": response.usage.get("prompt_tokens"),
            "completion_tokens": response.usage.get("completion_tokens"),
            "total_tokens": response.usage.get("total_tokens"),

            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_time": (end_time - start_time).total_seconds(),

            "response": dict(response),
        }

        # Log and save response information
        logging.info(f"API response: {response_dict}")
        self._responses.append(response_dict)

    @property
    def details(self) -> dict[str, Any]:
        """Get the details of the API usage tracker."""
        return {
            "total_cost": self.total_cost,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "num_api_calls": self.num_api_calls,
            "mean_response_time": self.mean_response_time,
            "response_times": {
                f"{p}percentile": f"{resp_time:.3f}"
                for p in [50, 75, 90, 95, 99, 99.9]
                if (resp_time := self.response_time_at_percentile(p)) is not None
            },
        }

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_prompt_tokens(self) -> int:
        """Total number of prompt tokens used across all API calls."""
        return sum(r['prompt_tokens'] for r in self._responses)

    @property
    def total_completion_tokens(self) -> int:
        """Total number of completion tokens used across all API calls."""
        return sum(r['completion_tokens'] for r in self._responses)

    @property
    def num_api_calls(self) -> int:
        """Number of API calls; or, more specifically, number of API responses."""
        return len(self._responses)

    @property
    def mean_response_time(self) -> float | None:
        """Mean response time of API calls in seconds."""
        return float(np.mean([r['elapsed_time'] for r in self._responses])) if self._responses else None

    def response_time_at_percentile(self, percentile: float) -> float | None:
        """Response time at a given percentile in seconds."""
        if not self._responses:
            return None

        return float(np.percentile(
            [r['elapsed_time'] for r in self._responses],
            percentile,
        ))

    def track_cost_callback(
        self,
        kwargs,
        completion_response,
        start_time,
        end_time,
    ):
        """Function to track cost of API calls.

        This function should be added as a callback to the litellm package, eg:
        ```py
        import litellm
        litellm.success_callback = [self.track_cost_callback]
        ```
        """
        try:
            # Get response cost in USD
            response_cost = kwargs.get("response_cost", 0)
            self._total_cost += response_cost

            # Log completion response
            self._log_response(
                response=completion_response,
                start_time=start_time,
                end_time=end_time,
            )
            logging.info(f"API call cost: {response_cost}; Total cost: {self._total_cost}")

        except Exception as e:
            logging.error(f"Failed to track cost of API calls: {e}")

    def set_up_litellm_cost_tracking(self):
        """Set up cost tracking for API calls using LiteLLM."""
        import litellm
        litellm.success_callback = [self.track_cost_callback]

    def get_stats_str(self) -> str:
        """Get a string representation of the API usage tracker."""
        response_times_str = ""
        if self._responses:
            response_times_str = (
                f"Mean response time: {self.mean_response_time:.2f} seconds\n"
                f"Response time at 99th percentile: {self.response_time_at_percentile(99):.3f} seconds\n"
            )

        return (
            f"Total cost of API calls: ${self.total_cost:.2f}\n"
            f"Total prompt tokens: {self.total_prompt_tokens}\n"
            f"Total completion tokens: {self.total_completion_tokens}\n"
            f"Number of responses: {len(self._responses)}\n"
            f"{response_times_str}"
        )

    def __str__(self):
        """String representation of the API usage tracker."""
        return self.get_stats_str()

    def __del__(self):
        """Destructor that prints stats when the object is being destroyed."""
        logging.info(self.get_stats_str())
