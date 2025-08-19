"""A helper class to run rate-limited API requests concurrently using threads."""

import os
import copy
import threading
from typing import Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import logging

import openai
import litellm
from pyrate_limiter import Limiter, Rate, Duration

from .api_tracker import APIUsageTracker
from ._params import ALL_COMPLETION_PARAMS


# OpenAI API Tier 4 has a rate limit of 10K RPM and 2M-10M TPM
OPENAI_API_REQUESTS_PER_MINUTE = 10_000
OPENAI_API_TOKENS_PER_MINUTE = 2_000_000

# Default max context window tokens
DEFAULT_MAX_CONTEXT_TOKENS_ENV_VAR = "DEFAULT_MAX_CONTEXT_TOKENS"
try:
    DEFAULT_MAX_CONTEXT_TOKENS = int(os.getenv(DEFAULT_MAX_CONTEXT_TOKENS_ENV_VAR, "20000"))
except ValueError:
    logging.getLogger(__name__).warning(
        f"Environment variable {DEFAULT_MAX_CONTEXT_TOKENS_ENV_VAR} must be an integer. "
        "Falling back to 20,000 tokens.")
    DEFAULT_MAX_CONTEXT_TOKENS = 20_000


class APIClient:
    """A generic API client to run rate-limited requests concurrently using threads.

    By default, uses the LiteLLM completion API.
    """

    def __init__(
        self,
        max_requests_per_minute: int = OPENAI_API_REQUESTS_PER_MINUTE,
        max_tokens_per_minute: int = OPENAI_API_TOKENS_PER_MINUTE,
        max_workers: int = None,
    ):
        """Initialize the API client.

        Parameters
        ----------
        max_requests_per_minute : int, optional
            Maximum API requests allowed per minute. Default is OPENAI_API_RPM.
        max_tokens_per_minute : int, optional
            Maximum tokens allowed per minute.
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
            "max_delay": 120_000,  # 2 minutes
        }

        # RPM limiter
        self._rpm_limiter = None
        if self.max_requests_per_minute is not None:
            requests_limit = Rate(self.max_requests_per_minute, Duration.MINUTE)
            self._rpm_limiter = Limiter(requests_limit, **limiter_config)

        # TPM limiter
        self._tpm_limiter = None
        if self.max_tokens_per_minute is not None:
            tokens_limit = Rate(self.max_tokens_per_minute, Duration.MINUTE)
            self._tpm_limiter = Limiter(tokens_limit, **limiter_config)

        # Logger
        self._logger = logging.getLogger(__name__)
        self._logged_msgs = set()

        # Remove handlers from litellm logger so messages propagate to root logger
        litellm_logger = logging.getLogger("LiteLLM")
        litellm_logger.handlers.clear()

        # Usage tracker
        self._tracker = APIUsageTracker()
        self._tracker.set_up_litellm_cost_tracking()

        # History of requests and responses
        self._history: list[dict] = []

    @property
    def details(self) -> dict[str, Any]:
        return {
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "max_workers": self._max_workers,
            **self._tracker.details,
        }

    @property
    def tracker(self) -> APIUsageTracker:
        """The API usage tracker instance."""
        return self._tracker

    @property
    def history(self) -> list[dict]:
        """The history of requests and responses."""
        return self._history

    def make_requests(
        self,
        requests: list[dict],
        *,
        max_workers: int = None,
        sanitize: bool = True,
        timeout: float = None,
    ) -> list[object]:
        """Make a series of rate-limited API requests concurrently using threads."""
        # Short-circuit: no requests
        if not requests:
            return []

        responses = [None] * len(requests)

        # Override max_workers if provided
        max_workers = min(max_workers or self._max_workers, len(requests))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_request_idx = {
                executor.submit(self._rate_limited_request, request=request, sanitize=sanitize): idx
                for idx, request in enumerate(requests)
            }

            try:
                for future in as_completed(future_to_request_idx, timeout=timeout):
                    request_idx = future_to_request_idx[future]
                    try:
                        request_response = future.result()
                        responses[request_idx] = request_response
                    except openai.APIError as e:
                        status_code = getattr(e, 'status_code', None)
                        message = getattr(e, 'message', None)
                        llm_provider = getattr(e, 'llm_provider', None)
                        self._logger.error(
                            "Request generated an APIError: status_code=%s; message=%s; llm_provider=%s",
                            status_code, message, llm_provider)
                        responses[request_idx] = None
                    except Exception as e:
                        self._logger.error("Request generated an exception: %s", e)
                        responses[request_idx] = None
            except FuturesTimeoutError as e:
                incomplete_indices = [idx for idx, resp in enumerate(responses) if resp is None]
                self._logger.error(
                    "Timeout reached after %s seconds. %d/%d requests did not complete; error=%s",
                    timeout, len(incomplete_indices), len(requests), e)

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
        """Make a series of rate-limited API requests with automatic retries for failed requests."""
        if current_retry > max_retries:
            self._logger.error(
                f"Exceeded max_retries ({max_retries}) for {len(requests)} requests; returning None responses")
            return [None] * len(requests)

        responses = self.make_requests(
            requests=requests,
            max_workers=max_workers,
            sanitize=sanitize,
            timeout=timeout,
        )

        failed_requests = []
        failed_requests_og_indices = []
        for idx, response in enumerate(responses):
            if response is None:
                self._logger.warning(
                    f"Request with idx={idx} failed; will be retried; Current retry: {current_retry + 1}/{max_retries};")
                failed_requests.append(requests[idx])
                failed_requests_og_indices.append(idx)

        if failed_requests:
            failed_requests_responses = self.make_requests_with_retries(
                requests=failed_requests,
                max_workers=max_workers,
                max_retries=max_retries,
                sanitize=sanitize,
                timeout=timeout,
                current_retry=current_retry + 1,
            )

            for idx_in_failed_requests, idx_in_original_requests in enumerate(failed_requests_og_indices):
                responses[idx_in_original_requests] = failed_requests_responses[idx_in_failed_requests]

        return responses

    def sanitize_completion_request(self, request: dict) -> dict:
        """Sanitize the request parameters for the completion API."""
        sanitized_request = self.remove_unsupported_params(request)
        sanitized_request["messages"] = self.truncate_to_max_context_tokens(
            messages=sanitized_request["messages"],
            model=sanitized_request["model"],
        )
        return sanitized_request

    def remove_unsupported_params(self, request: dict) -> dict:
        """Ensure request params are compatible with the model and provider."""
        request = copy.deepcopy(request)
        model = request.pop("model")
        messages = request.pop("messages")

        supported_params = litellm.get_supported_openai_params(
            model=model,
            request_type="chat_completion",
        )

        model_specific_unsupported_params = set()
        if model.lower().startswith("openai/o"):
            model_specific_unsupported_params = {"temperature"}

        supported_params = [p for p in supported_params if p not in model_specific_unsupported_params]

        supported_kwargs = {k: v for k, v in request.items() if k in supported_params}
        provider_specific_kwargs = {k: v for k, v in request.items() if k not in ALL_COMPLETION_PARAMS}

        if provider_specific_kwargs:
            msg = f"Provider-specific parameters for model='{model}' in API request: {provider_specific_kwargs}."
            if msg not in self._logged_msgs:
                self._logger.info(msg)
                self._logged_msgs.add(msg)

        unsupported_kwargs = {k: v for k, v in request.items() if (k not in supported_params and k not in provider_specific_kwargs)}
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
        """Truncate a prompt to the maximum context tokens for a model."""
        messages = copy.deepcopy(messages)
        max_tokens = self.get_max_context_tokens(model)

        def total_chars(msgs: list[dict]) -> int:
            return sum(len(m["content"]) for m in msgs)

        request_tok_len = self.count_messages_tokens(messages, model=model)
        request_char_len = total_chars(messages)

        while request_tok_len > max_tokens and messages:
            chars_per_token = max(1.0, request_char_len / max(request_tok_len, 1))
            tokens_to_drop = max(1, request_tok_len - max_tokens)
            approx_chars_to_drop = int(tokens_to_drop * chars_per_token)

            max_step = max(1, int(request_char_len * 0.15))
            num_chars_to_drop = min(approx_chars_to_drop, max_step)

            drop_fraction = num_chars_to_drop / max(request_char_len, 1)
            if drop_fraction > 0.5:
                self._logger.warning(
                    f"Dropping {drop_fraction} of the message due to token limit; "
                    f"request_char_len={request_char_len}, max_tokens={max_tokens}, request_tok_len={request_tok_len};")

            # Prefer trimming non-system messages; if none, trim last message
            idx = len(messages) - 1
            while idx >= 0 and messages[idx].get("role") == "system":
                idx -= 1
            if idx < 0:
                idx = len(messages) - 1

            remaining = num_chars_to_drop
            while remaining > 0 and messages:
                current_len = len(messages[idx]["content"])
                if current_len > remaining:
                    messages[idx]["content"] = messages[idx]["content"][: -remaining]
                    remaining = 0
                else:
                    remaining -= current_len
                    messages.pop(idx)
                    if not messages:
                        break
                    idx = len(messages) - 1
                    while idx >= 0 and messages[idx].get("role") == "system":
                        idx -= 1
                    if idx < 0:
                        idx = len(messages) - 1

            request_tok_len = self.count_messages_tokens(messages, model=model)
            request_char_len = total_chars(messages)

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
                f"Could not get max context tokens from litellm: {e}. Using fallback default of {DEFAULT_MAX_CONTEXT_TOKENS} tokens.")
            return DEFAULT_MAX_CONTEXT_TOKENS

    def count_messages_tokens(
        self,
        messages: list[dict],
        *,
        model: str,
        timeout: float = 10,
    ) -> int:
        """Count tokens in text using the model's tokenizer."""
        msgs_char_len = sum(
            len(msg["content"]) for msg in messages if (msg is not None and "content" in msg and msg["content"] is not None)
        )

        self._logger.debug(
            f"Counting tokens for model={model} and messages of char length={msgs_char_len}")

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(litellm.token_counter, model=model, messages=messages)
                return future.result(timeout=timeout)
        except FuturesTimeoutError:
            self._logger.error(
                f"Token counting timed out after {timeout} seconds. Using a rough approximation.")
            return msgs_char_len // 3
        except TimeoutError:
            self._logger.error(
                f"Token counting timed out after {timeout} seconds. Using a rough approximation.")
            return msgs_char_len // 3
        except Exception as e:
            self._logger.warning(
                f"Could not count tokens using litellm: {e}. Using a rough approximation.")
            return msgs_char_len // 3

    def _rate_limited_request(
        self,
        request: dict,
        sanitize: bool = True,
    ) -> object:
        """Submit an API call with the given parameters, honoring rate limits."""
        if sanitize:
            request = self.sanitize_completion_request(request)

        self._acquire_rate_limited_resources(request=request)

        self._logger.debug(
            f"Thread {threading.get_ident()} making API call with parameters: {request}")

        try:
            response = self.api_call(**request)
            self._logger.debug(
                f"Thread {threading.get_ident()} completed API call with response: {response}")
        except Exception as e:
            self._logger.error(
                f"Thread {threading.get_ident()} failed API call with error: {e}")
            raise e

        return response

    def _acquire_rate_limited_resources(self, *, request: dict) -> None:
        """Wait for the rate limit to be available and acquire necessary resources."""
        model = request.get("model")
        messages = request.get("messages", [])
        token_count = self.count_messages_tokens(messages, model=model)

        thread_id = threading.get_ident()
        self._logger.debug(
            f"Thread {thread_id} waiting for rate limit: request={1}, tokens={token_count}")

        if self._rpm_limiter is None and self._tpm_limiter is None:
            return

        import time as _time

        # Acquire RPM first (single unit)
        if self._rpm_limiter is not None:
            while True:
                try:
                    if self._rpm_limiter.try_acquire("api_calls", 1):
                        break
                except Exception:
                    pass
                _time.sleep(1)

        # Acquire TPM tokens one-by-one to avoid bucket overflow with large weights
        if self._tpm_limiter is not None and token_count > 0:
            tokens_remaining = token_count
            while tokens_remaining > 0:
                try:
                    if self._tpm_limiter.try_acquire("tokens", 1):
                        tokens_remaining -= 1
                        continue
                except Exception:
                    pass
                _time.sleep(1)

        self._logger.debug(
            f"Thread {thread_id} acquired rate limit resources: request={1}, tokens={token_count}")

    def __save_history(self, *, requests: list[dict], responses: list[object]) -> None:
        """Save API requests and responses to the client's history."""
        def get_response_dict(response):
            return getattr(response, "model_dump", lambda: response.__dict__)()

        def get_response_datetime(response):
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

