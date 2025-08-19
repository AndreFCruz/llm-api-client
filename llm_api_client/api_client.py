"""A helper class to run rate-limited API requests concurrently using threads.
"""
import os
import copy
import threading
from typing import Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import openai
import litellm
from pyrate_limiter import Limiter, Rate, Duration

from .api_tracker import APIUsageTracker  # Integrated API usage tracker
from ._params import ALL_COMPLETION_PARAMS

# OpenAI API Tier 4 has a rate limit of 10K RPM and 2M-10M TPM
OPENAI_API_REQUESTS_PER_MINUTE = 10_000
# OPENAI_API_TOKENS_PER_MINUTE = 10_000_000
OPENAI_API_TOKENS_PER_MINUTE = 2_000_000

# Default max context window tokens
DEFAULT_MAX_CONTEXT_TOKENS_ENV_VAR = "DEFAULT_MAX_CONTEXT_TOKENS"
try:
    DEFAULT_MAX_CONTEXT_TOKENS = int(os.getenv(DEFAULT_MAX_CONTEXT_TOKENS_ENV_VAR, "20000"))
except ValueError:
    logging.getLogger(__name__).warning(
        f"Environment variable {DEFAULT_MAX_CONTEXT_TOKENS_ENV_VAR} must be an integer. "
        "Falling back to 20,000 tokens.")
    DEFAULT_MAX_CONTEXT_TOKENS = 20_000  # 20K tokens context window (default)


class APIClient:
    """A generic API client to run rate-limited requests concurrently using threads.

    By default, uses the LiteLLM completion API.

    API requests and responses are logged and can optionally be saved to disk if a log
    file is specified. An APIUsageTracker instance is automatically instantiated to track
    the cost and usage of API calls.

    Examples
    --------
    >>> completion_api_client = APIClient(max_requests_per_minute=5)
    >>> requests = [
    >>>     dict(
    >>>         model="gpt-3.5-turbo",
    >>>         messages=[{"role": "user", "content": prompt}],
    >>>     ) for prompt in user_prompts
    >>> ]
    >>> results = completion_api_client.make_requests(requests)
    """

    def __init__(
        self,
        max_requests_per_minute: int = OPENAI_API_REQUESTS_PER_MINUTE,
        max_tokens_per_minute: int = OPENAI_API_TOKENS_PER_MINUTE,
        max_workers: int = None,
    ):
        """
        Parameters
        ----------
        max_requests_per_minute : int, optional
            Maximum API requests allowed per minute. Default is OPENAI_API_RPM.
        max_tokens_per_minute : int, optional
            Maximum tokens allowed per minute. Default is None (no token limit).
        max_workers : int, optional
            Maximum number of worker threads. Default is min(CPU count * 20, max_rpm).
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.api_call = litellm.completion

        # Get max_workers from CPU count
        self._max_workers = max_workers
        if self._max_workers is None and self.max_requests_per_minute is not None:
            self._max_workers = self.max_requests_per_minute
        if self._max_workers is None:
            self._max_workers = os.cpu_count() * 20

        # Set up rate limiter using pyrate-limiter
        limiter_config = {
            "max_delay": 120_000,    # 2 minutes
        }

        # Set up RPM limiter
        self._rpm_limiter = None
        if self.max_requests_per_minute is not None:
            requests_limit = Rate(self.max_requests_per_minute, Duration.MINUTE)
            self._rpm_limiter = Limiter(requests_limit, **limiter_config)

        # Set up TPM limiter
        self._tpm_limiter = None
        if self.max_tokens_per_minute is not None:
            tokens_limit = Rate(self.max_tokens_per_minute, Duration.MINUTE)
            self._tpm_limiter = Limiter(tokens_limit, **limiter_config)

        # Set up logger
        self._logger = logging.getLogger(__name__)
        self._logged_msgs = set()   # To avoid logging duplicate messages over and over

        # Remove handlers from litellm logger so messages propagate to root logger
        # > This way all LiteLLM logs will follow the settings of the root logger
        # > (e.g. log level, format, etc.)
        litellm_logger = logging.getLogger("LiteLLM")
        litellm_logger.handlers.clear()

        # Create an APIUsageTracker instance and set it up with LiteLLM.
        self._tracker = APIUsageTracker()
        self._tracker.set_up_litellm_cost_tracking()

        # History of requests and responses
        self._history: list[dict] = []

    @property
    def details(self) -> dict[str, Any]:
        """Get the details of the API client."""
        return {
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "max_workers": self._max_workers,
            **self._tracker.details,
        }

    @property
    def tracker(self) -> APIUsageTracker:
        """
        The API usage tracker instance.

        Returns
        -------
        llm_api_client.api_tracker.APIUsageTracker
            The API usage tracker.
        """
        return self._tracker

    @property
    def history(self) -> list[dict]:
        """
        The history of requests and responses.

        Returns
        -------
        list[dict]
            A list of request/response entries.
        """
        return self._history

    def make_requests(
        self,
        requests: list[dict],
        *,
        max_workers: int = None,
        sanitize: bool = True,
        timeout: float = None,
    ) -> list[object]:
        """Make a series of rate-limited API requests concurrently using threads.

        Parameters
        ----------
        requests : list[dict]
            A list of dictionaries, each containing the parameters to pass to
            the API function call.
        max_workers : int, optional
            The maximum number of threads to use in the ThreadPoolExecutor.
            If not provided, will default to: min(CPU count * 20, max_rpm).
        sanitize : bool, optional
            Whether to sanitize the requests; i.e., filter out request
            parameters that may be incompatible with the model and provider.
            Default is True.
        timeout : float, optional
            Maximum number of seconds to wait for all requests to complete.
            If None (default), waits indefinitely.

        Returns
        -------
        responses : list[object]
            A list of response objects returned by the API function calls.
            If a request fails, the corresponding response will be None.
        """
        # Create a list to store results, maintaining the order of requests
        responses = [None] * len(requests)

        # Override max_workers if provided
        max_workers = min(max_workers or self._max_workers, len(requests))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all requests to the executor
            future_to_request_idx = {
                executor.submit(self._rate_limited_request, request=request, sanitize=sanitize): idx
                for idx, request in enumerate(requests)
            }

            try:
                # Collect results as they complete
                for future in as_completed(future_to_request_idx, timeout=timeout):
                    request_idx = future_to_request_idx[future]

                    try:
                        request_response = future.result()
                        # Store responses in the same order as the requests
                        responses[request_idx] = request_response

                    # Handle generic API errors
                    except openai.APIError as e:
                        status_code = getattr(e, 'status_code', None)
                        message = getattr(e, 'message', None)
                        llm_provider = getattr(e, 'llm_provider', None)
                        self._logger.error(
                            "Request generated an APIError: "
                            "status_code=%s; message=%s; llm_provider=%s",
                            status_code, message, llm_provider)
                        responses[request_idx] = None   # Explicitly set failed responses to None

                    # Catch all other exceptions thrown by child threads
                    except Exception as e:
                        self._logger.error("Request generated an exception: %s", e)
                        responses[request_idx] = None   # Explicitly set failed responses to None

            # Handle timeout from as_completed
            except TimeoutError as e:
                incomplete_indices = [idx for idx, resp in enumerate(responses) if resp is None]
                self._logger.error(
                    "Timeout reached after %s seconds. %d/%d requests did not complete; "
                    "error=%s",
                    timeout, len(incomplete_indices), len(requests), e)
                # Note: responses array already initialized with None values for incomplete requests

        # Save all responses to API client history
        self.__save_history(requests=requests, responses=responses)

        return responses

    def make_requests_with_retries(
        self,
        requests: list[dict],
        *,
        max_workers: int = None,
        max_retries: int = 2,
        sanitize: bool = True,
        timeout: float = None,
        current_retry: int = 0,
    ) -> list[object]:
        """Make a series of rate-limited API requests with automatic retries for failed requests.

        Parameters
        ----------
        requests : list[dict]
            A list of dictionaries, each containing the parameters to pass to the API function call.
        max_workers : int, optional
            Maximum number of worker threads to use.
        max_retries : int, optional
            Maximum number of retry attempts for failed requests.
        sanitize : bool, optional
            Whether to sanitize the request parameters.
        timeout : float, optional
            Maximum number of seconds to wait for all requests to complete.
            If None (default), waits indefinitely.
        current_retry : int, optional
            Current retry attempt number (used internally for recursion).

        Returns
        -------
        list[object]
            A list of response objects returned by the API function calls.
        """
        # **Break case**: Check if we've exceeded max_retries
        if current_retry > max_retries:
            self._logger.error(
                f"Exceeded max_retries ({max_retries}) for {len(requests)} requests; "
                "returning None responses for all requests")
            return [None] * len(requests)

        # **Base case**: Make the API calls
        responses = self.make_requests(
            requests=requests,
            max_workers=max_workers,
            sanitize=sanitize,
            timeout=timeout,
        )

        # **Recursive case**: Gather failed requests and retry them
        failed_requests = []
        failed_requests_og_indices = []
        for idx, response in enumerate(responses):
            if response is None:
                self._logger.warning(
                    f"Request with idx={idx} failed; will be retried; "
                    f"Current retry: {current_retry + 1}/{max_retries};")

                failed_requests.append(requests[idx])
                failed_requests_og_indices.append(idx)

        # If there are failed requests, retry them
        if failed_requests:
            # Recursively retry failed requests
            failed_requests_responses = self.make_requests_with_retries(
                requests=failed_requests,
                max_workers=max_workers,
                max_retries=max_retries,
                sanitize=sanitize,
                timeout=timeout,
                current_retry=current_retry + 1,
            )

            # Update the responses in the order of the original requests
            for idx_in_failed_requests, idx_in_original_requests in enumerate(failed_requests_og_indices):
                responses[idx_in_original_requests] = failed_requests_responses[idx_in_failed_requests]

        return responses

    def sanitize_completion_request(self, request: dict) -> dict:
        """Sanitize the request parameters for the completion API.

        1. Checks and removes unsupported parameters for this model and provider.
        2. Truncates the request to the maximum context tokens for the model.

        Returns
        -------
        sanitized_request : dict
            A dictionary containing parsed and filtered request parameters.
        """
        # 1. Remove unsupported parameters
        sanitized_request = self.remove_unsupported_params(request)

        # 2. Truncate the request to the maximum context tokens for the model
        sanitized_request["messages"] = self.truncate_to_max_context_tokens(
            messages=sanitized_request["messages"],
            model=sanitized_request["model"],
        )

        return sanitized_request

    def remove_unsupported_params(self, request: dict) -> dict:
        """Ensure request params are compatible with the model and provider.

        Checks and removes unsupported parameters for this model and provider.

        Returns
        -------
        compatible_request : dict
            A dictionary containing the provided request with all unsupported
            parameters removed.
        """
        # Make a copy of the request to avoid mutating the original
        request = copy.deepcopy(request)

        # Extract model and messages from request
        model = request.pop("model")
        messages = request.pop("messages")

        # Sanitize general provider params
        supported_params = litellm.get_supported_openai_params(
            model=model,
            request_type="chat_completion",
        )

        # Hardcoding some model-specific params
        model_specific_unsupported_params = set()
        if model.lower().startswith("openai/o"):
            model_specific_unsupported_params = {"temperature"}

        # Eliminate model-specific unsupported params
        supported_params = [
            p for p in supported_params
            if p not in model_specific_unsupported_params
        ]

        # Construct dict of supported kwargs
        supported_kwargs = {k: v for k, v in request.items() if k in supported_params}

        # Check provider-specific params (any non-openai and non-litellm params)
        provider_specific_kwargs = {k: v for k, v in request.items() if k not in ALL_COMPLETION_PARAMS}

        # Log provider-specific kwargs as info
        if provider_specific_kwargs:
            msg = f"Provider-specific parameters for model='{model}' in API request: {provider_specific_kwargs}."
            if msg not in self._logged_msgs:
                self._logger.info(msg)
                self._logged_msgs.add(msg)

        # Log unsupported kwargs
        unsupported_kwargs = {
            k: v for k, v in request.items()
            if (k not in supported_params and k not in provider_specific_kwargs)
        }
        if unsupported_kwargs:
            msg = f"Unsupported parameters for model='{model}' in API request: {unsupported_kwargs}."
            if msg not in self._logged_msgs:
                self._logger.error(msg)
                self._logged_msgs.add(msg)

        return {"model": model, "messages": messages, **supported_kwargs, **provider_specific_kwargs}

    def truncate_to_max_context_tokens(
        self,
        messages: list[dict],
        model: str,
    ) -> list[dict]:
        """Truncate a prompt to the maximum context tokens for a model.

        Parameters
        ----------
        messages : list[dict]
            The request messages to truncate.
        model : str
            The name of the model to use.

        Returns
        -------
        list[dict]
            The request messages, truncated so that the total token count is
            less than or equal to the maximum context tokens for the model.
        """
        messages = copy.deepcopy(messages)
        request_char_len = sum(len(msg["content"]) for msg in messages)
        request_tok_len = self.count_messages_tokens(messages, model=model)
        max_tokens = self.get_max_context_tokens(model)

        # Easy heuristic: iteratively drop the last 10% of the prompt
        while request_tok_len > max_tokens:
            # Estimate the ratio of chars to keep by comparing request_tok_len and max_tokens
            ratio_of_chars_to_keep = min(
                0.9,  # always drop at least 10%
                (max_tokens / request_tok_len) - 0.05,
            )

            # Log a warning if we're dropping more than 50% of the message
            if ratio_of_chars_to_keep < 0.5:
                self._logger.warning(
                    f"Dropping {1 - ratio_of_chars_to_keep} of the message due to "
                    f"token limit; request_char_len={request_char_len}, "
                    f"max_tokens={max_tokens}, request_tok_len={request_tok_len};")

            # Ensure we always keep at least 5% of the message
            ratio_of_chars_to_keep = max(0.05, ratio_of_chars_to_keep)
            assert 0 < ratio_of_chars_to_keep < 1

            # Compute number of chars to drop
            num_chars_to_drop = int(request_char_len * (1 - ratio_of_chars_to_keep))
            assert num_chars_to_drop > 0

            # Drop `num_chars_to_drop` chars from the last message
            if len(messages[-1]["content"]) > num_chars_to_drop:
                messages[-1]["content"] = messages[-1]["content"][: -num_chars_to_drop]
            else:
                # If the last message is shorter than `num_chars_to_drop`, drop the entire message
                messages.pop()

                self._logger.warning(
                    f"Could not truncate {num_chars_to_drop} chars from the last message; "
                    f"dropping entire message instead...")

            # Re-count tokens
            request_tok_len = self.count_messages_tokens(messages, model=model)
            request_char_len = sum(len(msg["content"]) for msg in messages)

        return messages

    def get_max_context_tokens(self, model: str) -> int:
        """Get the maximum context tokens for a model."""
        try:
            model_info = litellm.get_model_info(model)
            max_tokens = model_info.get("max_input_tokens")
            if max_tokens is None:
                raise ValueError("max_input_tokens not provided by litellm")
            return max_tokens
        except Exception as e:
            self._logger.warning(
                f"Could not get max context tokens from litellm: {e}. "
                f"Using fallback default of {DEFAULT_MAX_CONTEXT_TOKENS} tokens.")
            return DEFAULT_MAX_CONTEXT_TOKENS

    def count_messages_tokens(
        self,
        messages: list[dict],
        *,
        model: str,
        timeout: float = 10,
    ) -> int:
        """Count tokens in text using the model's tokenizer.

        Parameters
        ----------
        messages : list[dict]
            The messages to count tokens for.
        model : str
            The model to count tokens for.
        timeout : float, optional
            The timeout for the token counting operation in seconds.

        Returns
        -------
        int
            The number of tokens in the messages.
        """
        msgs_char_len = sum(
            len(msg["content"])
            for msg in messages
            if (msg is not None and "content" in msg and msg["content"] is not None)
        )

        self._logger.debug(
            f"Counting tokens for model={model} and messages of char length={msgs_char_len}")

        try:
            # Use ThreadPoolExecutor to run token counting with a timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(litellm.token_counter, model=model, messages=messages)
                return future.result(timeout=timeout)

        except TimeoutError:
            self._logger.error(
                f"Token counting timed out after {timeout} seconds. Using a rough approximation.")
            return msgs_char_len // 3  # Rough approximation

        except Exception as e:
            self._logger.warning(
                f"Could not count tokens using litellm: {e}. Using a rough approximation.")
            return msgs_char_len // 3  # Rough approximation

    def _rate_limited_request(
        self,
        request: dict,
        sanitize: bool = True,
    ) -> object:
        """Submit an API call with the given parameters, honoring rate limits.

        Will block the calling thread until the request can be made without violating
        the rate limits!

        Parameters
        ----------
        request : dict
            The request parameters to pass to the API function call.
        sanitize : bool, optional
            Whether to sanitize the request; i.e., parse and filter out request
            parameters that may be incompatible with the model and provider.
            Default is True.

        Returns
        -------
        response : object
            The response object returned by the API function call.
        """
        # Sanitize request if requested
        if sanitize:
            request = self.sanitize_completion_request(request)

        # Block until the rate limit is available
        self._acquire_rate_limited_resources(request=request)

        # Log the API request parameters
        self._logger.debug(
            f"Thread {threading.get_ident()} making API call with "
            f"parameters: {request}")

        try:
            # Make the API call
            response = self.api_call(**request)
            self._logger.debug(
                f"Thread {threading.get_ident()} completed API call with "
                f"response: {response}")

        except Exception as e:
            self._logger.error(
                f"Thread {threading.get_ident()} failed API call with error: {e}")
            raise e

        return response

    def _acquire_rate_limited_resources(self, *, request: dict) -> None:
        """Wait for the rate limit to be available and acquire necessary resources.

        This function blocks the current thread until rate limits allow the request
        to proceed. It accounts for both request count and token count.

        Parameters
        ----------
        request : dict
            The API request parameters containing model and messages.
        """
        # Count tokens in the request to know how many token resources to consume
        model = request.get("model")
        messages = request.get("messages", [])

        # Get token count for this request
        token_count = self.count_messages_tokens(messages, model=model)

        # Log that we're waiting for rate limit
        thread_id = threading.get_ident()
        self._logger.debug(
            f"Thread {thread_id} waiting for rate limit: "
            f"request={1}, tokens={token_count}"
        )

        # Block until we can acquire the rate limit resources
        # NOTE: Acquiring each resource separately can lead to a RACE CONDITION
        # where we acquire the RPM but not the TPM, or vice versa.
        # However, it won't lead to a DEADLOCK since resources get released over
        # time -- but it will lead to suboptimal resource utilization.
        if self._rpm_limiter is not None:
            self._rpm_limiter.try_acquire("api_calls", weight=1)
        if self._tpm_limiter is not None:
            self._tpm_limiter.try_acquire("tokens", weight=token_count)

        self._logger.debug(
            f"Thread {thread_id} acquired rate limit resources: "
            f"request={1}, tokens={token_count}"
        )

    def __save_history(self, *, requests: list[dict], responses: list[object]) -> None:
        """Save API requests and responses to the client's history.

        Parameters
        ----------
        requests : list[dict]
            The list of API request parameters.
        responses : list[object]
            The list of API responses corresponding to the requests.
        """
        def get_response_dict(response):
            return getattr(response, "model_dump", lambda: response.__dict__)()

        def get_response_datetime(response):
            """Convert a Unix timestamp to a formatted datetime string."""
            return datetime.fromtimestamp(response["created"]).strftime('%Y-%m-%d %H:%M:%S')

        def get_response_content(response):
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return None

        self._history.extend([
            {
                "request": request,
                "response": get_response_dict(response) if response else None,
                "content": get_response_content(response) if response else None,
                "created_at": get_response_datetime(response) if response else None,
            }
            for request, response in zip(requests, responses)
        ])
