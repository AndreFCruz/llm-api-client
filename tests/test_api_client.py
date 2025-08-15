"""
Test suite for APIClient.

These tests mock the litellm library to avoid making actual API calls.
"""
import time
import copy
from unittest.mock import patch

import pytest

from llm_api_client import APIClient

from mock_objects import MockResponse


class TestAPIClient:
    """Test suite for APIClient class."""

    @patch('litellm.completion')
    @patch('litellm.token_counter')
    def test_init(self, mock_token_counter, mock_completion):
        """Test initialization of APIClient."""
        max_rpm, max_tpm = 1000, 1000
        client = APIClient(max_requests_per_minute=max_rpm, max_tokens_per_minute=max_tpm)
        assert client.max_requests_per_minute == max_rpm
        assert client.max_tokens_per_minute == max_tpm
        assert client._history == []
        assert hasattr(client, '_tracker')

    @patch('litellm.completion')
    def test_make_single_request(self, mock_completion):
        """Test making a single request."""
        # Setup mocks
        mock_completion.return_value = MockResponse("Test response")

        # Initialize client and make request
        client = APIClient(max_requests_per_minute=5)
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        results = client.make_requests([request])

        # Assertions
        assert len(results) == 1
        assert results[0].choices[0].message.content == "Test response"
        mock_completion.assert_called_once()

        # Check history
        assert len(client.history) == 1
        assert client.history[0]["request"] == request
        assert client.history[0]["content"] == "Test response"

    @patch('litellm.completion')
    def test_multiple_requests(self, mock_completion):
        """Test making multiple requests (sequentially)."""
        n_requests = 3

        # Setup mocks
        mock_completion.side_effect = [
            MockResponse(f"Response {i}") for i in range(n_requests)
        ]

        # Initialize client and requests
        # NOTE: `max_workers=1` forces requests to be sent sequentially to match order of side_effects
        client = APIClient(max_requests_per_minute=5, max_workers=1)
        requests = [
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": f"Hello {i}"}]
            }
            for i in range(n_requests)
        ]

        # Submit requests
        results = client.make_requests(requests)

        # Assertions
        assert len(results) == n_requests
        assert mock_completion.call_count == n_requests
        for i, response in enumerate(results):
            assert response.choices[0].message.content == f"Response {i}"

        # Check history
        assert len(client.history) == n_requests

    @patch('litellm.completion')
    @patch('litellm.get_supported_openai_params')
    def test_sanitize_request(self, mock_get_params, mock_completion):
        """Test request sanitization."""
        # Setup mocks
        mock_completion.return_value = MockResponse()
        mock_get_params.return_value = ["temperature", "max_tokens"]

        # Initialize client
        client = APIClient()

        # Request with both supported and unsupported parameters
        request = {
            "model": "openai/o3-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "provider_specific_param": "value",  # This should be preserved as provider-specific
            "temperature": 0.7,                  # This should be removed as it's not supported by o3-mini
        }

        # Make request with sanitization
        _ = client.make_requests([request])

        # Check that:
        # - unsupported `temperature` param was removed when making the request
        # - provider-specific `provider_specific_param` was preserved
        sanitized_request = mock_completion.call_args[1]
        assert "temperature" not in sanitized_request           # Not compatible with o3-mini
        assert "provider_specific_param" in sanitized_request   # Unexpected kwargs are preserved as provider-specific
        assert "max_tokens" in sanitized_request
        assert "messages" in sanitized_request

    @patch('litellm.completion')
    @patch('litellm.token_counter')
    @patch('litellm.get_model_info')
    def test_truncate_messages(self, mock_get_model_info, mock_token_counter, mock_completion):
        """Test truncation of messages to fit context window."""
        # Setup mocks
        # First call returns tokens above limit, second call after truncation returns acceptable count
        mock_token_counter.side_effect = [1500, 800]
        mock_get_model_info.return_value = {"max_input_tokens": 1000}
        mock_completion.return_value = MockResponse()

        # Initialize client
        client = APIClient()

        # Request with messages that exceed the context window
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "A" * 5000}]  # Long message
        }

        # Make request
        _ = client.make_requests([request])

        # Verify the message was truncated
        request_with_truncated_content = mock_completion.call_args[1]
        original_content = request["messages"][0]["content"]
        truncated_content = request_with_truncated_content["messages"][0]["content"]
        assert len(original_content) > len(truncated_content)
        assert len(truncated_content) < 5000

    @patch('litellm.completion')
    def test_rate_limiting_requests_per_minute(self, mock_completion):
        """Test rate limiting RPM functionality."""
        # Setup mocks
        mock_completion.return_value = MockResponse()

        # Initialize client with max_RPM == 2
        client = APIClient(max_requests_per_minute=2, max_tokens_per_minute=None)

        # Prepare 3 requests
        requests = [
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": f"Test {i}"}]
            }
            for i in range(3)
        ]

        # Record start time
        start_time = time.time()

        # This should take extra time due to rate limiting
        results = client.make_requests(requests, max_workers=10, sanitize=False)

        # Measure elapsed time
        elapsed = time.time() - start_time

        # The 3rd request should be delayed for 60 seconds due to rate limiting
        assert 60 <= elapsed <= 70, "Rate limiting didn't seem to work as expected"
        assert len(results) == 3
        assert mock_completion.call_count == 3

    @patch('litellm.completion')
    @patch('litellm.token_counter')
    def test_rate_limiting_tokens_per_minute(self, mock_token_counter, mock_completion):
        """Test rate limiting TPM functionality."""
        n_requests = 2
        request_tokens = 1000
        max_tokens_per_minute = 1000

        # Setup mocks
        mock_completion.return_value = MockResponse()
        mock_token_counter.return_value = request_tokens

        # Initialize client with max_TPM == 1000
        client = APIClient(
            max_requests_per_minute=None,
            max_tokens_per_minute=max_tokens_per_minute,
        )

        # Prepare 3 requests
        requests = [
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": f"Test {i}"}]
            }
            for i in range(n_requests)
        ]

        # Record start time
        start_time = time.time()

        # This should take extra time due to rate limiting
        results = client.make_requests(requests, max_workers=10, sanitize=False)

        # Measure elapsed time
        elapsed = time.time() - start_time

        # The 2nd request should be delayed for 60 seconds due to rate limiting
        assert 60 <= elapsed <= 70, "Rate limiting didn't seem to work as expected"
        assert len(results) == n_requests
        assert mock_completion.call_count == n_requests

    @patch('litellm.completion')
    def test_request_failure_handling(self, mock_completion):
        """Test handling of failed requests."""
        # Make the second request fail
        mock_completion.side_effect = [
            MockResponse("Success"),
            Exception("API Error"),
            MockResponse("Success again")
        ]

        # Initialize client
        client = APIClient(max_workers=1)

        # Prepare requests
        requests = [
            {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": f"Request {i}"}]}
            for i in range(3)
        ]

        # Make requests
        results = client.make_requests(requests)

        # Check results - the second one should be None
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

        # Check history
        assert len(client.history) == 3
        assert client.history[0]["content"] == "Success"
        assert client.history[1]["content"] is None
        assert client.history[2]["content"] == "Success again"

    @patch('litellm.completion')
    def test_retries(self, mock_completion):
        """Test retry mechanism for failed requests."""
        # Failed request followed by success on retry
        mock_completion.side_effect = [
            Exception("API Error"),  # First attempt fails
            MockResponse("Retry success")  # Retry succeeds
        ]

        # Initialize client
        client = APIClient(max_workers=1)

        # Request that will fail on first attempt
        request = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        # Make request with retries - run a successful API call first to add data
        # This ensures we have the proper response structure
        with patch('litellm.completion', return_value=MockResponse("Warmup")):
            client.make_requests([request])

        # Reset mock for our actual test
        mock_completion.side_effect = [
            Exception("API Error"),  # First attempt fails
            MockResponse("Retry success")  # Retry succeeds
        ]

        # Make request with retries
        results = client.make_requests_with_retries([request], max_retries=1)

        # Should succeed on retry
        assert results[0] is not None

        # Access the model_dump() result as that's what make_requests_with_retries returns
        assert results[0].choices[0].message.content == "Retry success"
        assert mock_completion.call_count == 1 + 2  # 1 for the warmup request, 2 for the retries

    @patch('litellm.completion')
    def test_retries_many_requests(self, mock_completion):
        """Test retry mechanism for one failed requests among many.

        Verifies that when multiple requests are submitted and one fails,
        the retry mechanism correctly handles the failure and preserves
        the original request order in the results.
        """
        # Set the number of total requests
        n = 100
        n_retries = 10

        # Index of the request that will fail (middle of the list)
        fail_index = n // 2

        # Create mock responses for each request
        mock_responses = [
            MockResponse(f"Success {i+1}") for i in range(n)
        ]

        # Set up side effects - all requests succeed except for the one at fail_index
        side_effects = [
            mock_responses[i] if i != fail_index else Exception("API Error")
            for i in range(n)
        ]

        # Add the successful retried response at the end
        for i in range(n_retries):
            # Add failed responses for the first n_retries - 1 retries
            if i < n_retries - 1:
                side_effects.append(Exception("API Error"))

            # Add the successful response at the last retry
            else:
                side_effects.append(mock_responses[fail_index])

        # Set-up completion mock return values
        assert len(side_effects) == n + n_retries
        mock_completion.side_effect = side_effects

        # Initialize client
        client = APIClient(
            max_requests_per_minute=1_000_000,  # effectively no limit
            max_tokens_per_minute=None,
            max_workers=1,  # NOTE: Force requests to be sent sequentially to match order of side_effects
        )

        # Prepare requests programmatically
        requests = [
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": f"Request {i+1}"}]
            }
            for i in range(n)
        ]

        # Make requests with retries
        results = client.make_requests_with_retries(requests, max_retries=n_retries)

        # Verify results
        assert len(results) == n, "Should return same number of results as requests"

        # Check content of responses
        for i in range(n):
            assert results[i].choices[0].message.content == f"Success {i+1}"

        # Verify all requests were called (n initial + n retries)
        assert mock_completion.call_count == n + n_retries

    @patch('litellm.completion')
    @patch('litellm.token_counter')
    def test_details_property(self, mock_token_counter, mock_completion):
        """Test the details property returns expected information."""
        client = APIClient(max_requests_per_minute=10, max_tokens_per_minute=2000)

        # Make an actual request to ensure the tracker has complete data
        mock_completion.return_value = MockResponse("Test response")
        mock_token_counter.return_value = 50

        # Make a real request to populate response times
        client.make_requests([{
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        }])

        details = client.details
        assert details["max_requests_per_minute"] == 10
        assert details["max_tokens_per_minute"] == 2000
        assert "max_workers" in details

    @patch('litellm.completion')
    def test_timeout_error(self, mock_completion):
        """Test handling of request timeout."""
        # Configure the mock to sleep longer than our timeout
        def slow_api_call(*args, **kwargs):
            time.sleep(2)  # Sleep for 2 seconds
            return MockResponse("This response comes too late")

        mock_completion.side_effect = slow_api_call

        # Initialize client with normal settings
        client = APIClient(max_requests_per_minute=100, max_tokens_per_minute=None)

        # Prepare multiple requests so we can see which ones complete
        n_requests = 5
        requests = [
            {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": f"Request {i}"}]
            }
            for i in range(n_requests)
        ]

        # Set a very short timeout (0.5 seconds) that will cause the requests to timeout
        timeout = 0.5

        # Make requests with the short timeout
        results = client.make_requests(requests, timeout=timeout)

        # Verify all responses are None due to timeout
        assert all(response is None for response in results)
        assert len(results) == n_requests

        # Verify that the history contains the correct number of entries
        assert len(client.history) == n_requests
        # Verify that all content values in history are None
        assert all(entry["content"] is None for entry in client.history)

    @patch('litellm.token_counter')
    def test_count_messages_tokens(self, mock_token_counter):
        """Test the count_messages_tokens function."""
        # Setup
        mock_token_counter.return_value = 100
        client = APIClient()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        model = "gpt-3.5-turbo"

        # Test normal case
        token_count = client.count_messages_tokens(messages, model=model)
        assert token_count == 100
        mock_token_counter.assert_called_once_with(model=model, messages=messages)

        # Test timeout case
        mock_token_counter.reset_mock()
        mock_token_counter.side_effect = TimeoutError("Token counting timed out")

        # Calculate expected fallback token count (character count / 3)
        char_len = sum(len(msg["content"]) for msg in messages)
        expected_fallback_count = char_len // 3

        token_count = client.count_messages_tokens(messages, model=model, timeout=0.1)
        assert token_count == expected_fallback_count

        # Test exception case
        mock_token_counter.reset_mock()
        mock_token_counter.side_effect = Exception("Some error")

        token_count = client.count_messages_tokens(messages, model=model)
        assert token_count == expected_fallback_count

    @patch('litellm.completion')
    def test_truncate_to_max_context_tokens_direct(self, mock_completion):
        """Test direct call to truncate_to_max_context_tokens without mocking token counter.

        This test verifies that the method correctly truncates long messages to fit
        within context window by directly calling the method and checking if the
        token count is below the maximum after truncation.
        """
        # Initialize client
        client = APIClient()

        # Create a very long message
        original_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "a tree" * 10_000}  # Very long message
        ]

        # Make a copy of the original messages for comparison
        messages_copy = copy.deepcopy(original_messages)

        # Get initial token count using actual litellm counter
        initial_token_count = client.count_messages_tokens(messages_copy, model="gpt-3.5-turbo")

        # Get max tokens for the model using actual litellm get_model_info
        max_tokens = client.get_max_context_tokens("gpt-3.5-turbo")

        # Verify initial token count exceeds max tokens
        assert initial_token_count > max_tokens, \
            "Test setup issue: initial message not long enough to exceed token limit"

        # Call the truncation method directly
        truncated_messages = client.truncate_to_max_context_tokens(
            messages=messages_copy,
            model="gpt-3.5-turbo",
        )

        # Get final token count after truncation
        final_token_count = client.count_messages_tokens(truncated_messages, model="gpt-3.5-turbo")

        # Verify the token count is now within limits
        reasonable_lower_bound = int(max_tokens * 0.8)
        assert reasonable_lower_bound <= final_token_count <= max_tokens, (
            f"Truncated message from {initial_token_count} tokens to "
            f"{final_token_count} tokens; should be between "
            f"{reasonable_lower_bound} and {max_tokens}."
        )

        # Verify the user message was truncated
        assert len(truncated_messages[1]["content"]) < len(original_messages[1]["content"])

        # Verify the system message was not changed (we expect only the last message to be truncated)
        assert truncated_messages[0]["content"] == original_messages[0]["content"]

        # Verify the roles of the messages were preserved
        assert truncated_messages[0]["role"] == original_messages[0]["role"]
        assert truncated_messages[1]["role"] == original_messages[1]["role"]

    @patch('litellm.completion')
    def test_truncate_to_max_context_tokens_with_fixtures(self, mock_completion, test_messages):
        """Test truncate_to_max_context_tokens with various message configurations.

        Parameters
        ----------
        mock_completion : MagicMock
            Mock for litellm.completion
        test_messages : list
            Fixture providing test messages
        """
        # Initialize client
        client = APIClient()

        # Make a copy of the original messages for comparison
        messages_copy = copy.deepcopy(test_messages)

        # Get initial token count using actual litellm counter
        initial_token_count = client.count_messages_tokens(messages_copy, model="gpt-3.5-turbo")

        # Get max tokens for the model using actual litellm get_model_info
        max_tokens = client.get_max_context_tokens("gpt-3.5-turbo")

        # Skip the test if initial token count is not large enough to need truncation
        if initial_token_count <= max_tokens:
            pytest.skip(f"Test message with {initial_token_count} tokens doesn't exceed model limit of {max_tokens}")

        # Call the truncation method directly
        truncated_messages = client.truncate_to_max_context_tokens(
            messages=messages_copy,
            model="gpt-3.5-turbo",
        )

        # Get final token count after truncation
        final_token_count = client.count_messages_tokens(truncated_messages, model="gpt-3.5-turbo")

        # Verify the token count is now within limits
        reasonable_lower_bound = int(max_tokens * 0.8)
        assert reasonable_lower_bound <= final_token_count <= max_tokens, (
            f"Truncated message from {initial_token_count} tokens to "
            f"{final_token_count} tokens; should be between "
            f"{reasonable_lower_bound} and {max_tokens}."
        )

        # Verify the content was actually truncated
        total_original_content = sum(len(msg["content"]) for msg in test_messages)
        total_truncated_content = sum(len(msg["content"]) for msg in truncated_messages)
        assert total_truncated_content < total_original_content

        # Verify all messages still have role and content fields
        for msg in truncated_messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["system", "user", "assistant"]
            assert isinstance(msg["content"], str)


if __name__ == "__main__":
    pytest.main([__file__])
