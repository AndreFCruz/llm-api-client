# `llm-api-client` :robot::zap:

[![Docs status](https://github.com/AndreFCruz/llm-api-client/actions/workflows/docs.yml/badge.svg)](https://andrefcruz.github.io/llm-api-client/)
![Tests status](https://github.com/AndreFCruz/llm-api-client/actions/workflows/tests.yml/badge.svg)
![PyPI status](https://github.com/AndreFCruz/llm-api-client/actions/workflows/pypi-publish.yml/badge.svg)
![PyPI version](https://badgen.net/pypi/v/llm-api-client)
![PyPI - License](https://img.shields.io/pypi/l/llm-api-client)
![Python compatibility](https://badgen.net/pypi/python/llm-api-client)

A Python helper library for efficiently managing concurrent, rate-limited API requests to LLM providers via [LiteLLM](https://github.com/BerriAI/litellm).

It provides an `APIClient` that handles:
*   **Concurrency:** Making multiple API calls simultaneously using threads.
*   **Rate Limiting:** Respecting API limits for requests per minute (RPM) and tokens per minute (TPM).
*   **Retries:** Automatically retrying failed requests.
*   **Request Sanitization:** Cleaning up request parameters to ensure compatibility with different models/providers.
*   **LLM Context Management:** Truncating message history to fit within model context windows.
*   **Usage Tracking:** Monitoring API costs, token counts, and response times via an integrated `APIUsageTracker`.

Code documentation available at [https://andrefcruz.github.io/llm-api-client/](https://andrefcruz.github.io/llm-api-client/)

## Installation

Install the package directly from PyPI:

```bash
pip install llm-api-client
```

## Usage

The primary way to interact with the `APIClient` is through its `make_requests` and `make_requests_with_retries` methods, which handle concurrent execution, rate limiting, and retrying failed requests.

Here's a basic example of using `APIClient` to make multiple completion requests concurrently:

```python
import os
from llm_api_client import APIClient

# Ensure your API key is set (e.g., OPENAI_API_KEY environment variable)
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create a client with specific rate limits (adjust as needed)
# Defaults use OpenAI Tier 4 limits if not specified.
client = APIClient(
    max_requests_per_minute=1000,
    max_tokens_per_minute=100000
)

# Prepare your API requests
prompts = [
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about a cat.",
    "What is the capital of France?",
]

requests_data = [
    {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        # Add other parameters like temperature, max_tokens etc. if needed
        # "temperature": 0.7,
        # "max_tokens": 150,
    }
    for prompt in prompts
]

# Make the requests concurrently
# Use make_requests_with_retries for built-in retry logic
responses = client.make_requests(requests_data)

# Process the responses
for i, response in enumerate(responses):
    if response:
        # Access response content (structure depends on the API/model)
        # For OpenAI/LiteLLM completion:
        try:
            message_content = response.choices[0].message.content
            print(f"Response {i+1}: {message_content[:100]}...") # Print first 100 chars
        except (AttributeError, IndexError, TypeError) as e:
            print(f"Response {i+1}: Could not parse response content. Error: {e}")
            print(f"Raw response: {response}")
    else:
        print(f"Response {i+1}: Request failed.")

# Access usage statistics
print("\n--- Usage Statistics ---")
print(client.tracker) # Prints detailed stats

# Or access specific stats
print(f"Total cost: ${client.tracker.total_cost:.4f}")
print(f"Total prompt tokens: {client.tracker.total_prompt_tokens}")
print(f"Total completion tokens: {client.tracker.total_completion_tokens}")
print(f"Number of successful API calls: {client.tracker.num_api_calls}")
print(f"Mean response time: {client.tracker.mean_response_time:.2f}s")

# View request/response history
# print("\n--- History ---")
# for entry in client.history:
#     print(entry)
```

### Method Parameters

Both `make_requests` and `make_requests_with_retries` accept the following core parameters:

*   `requests` (list[dict]): A list where each dictionary represents the parameters for a single API call (e.g., `model`, `messages`, `temperature`, etc.) -- follows the openai API standard via [`litellm`](https://github.com/BerriAI/litellm).
*   `max_workers` (int, optional): The maximum number of concurrent threads to use for making API calls. Defaults to `min(CPU count * 20, max_rpm)`.
*   `sanitize` (bool, optional): If `True` (default), the client will attempt to remove parameters that are incompatible with the specified model and provider before making the request. It also truncates message history to fit the model's context window.
*   `timeout` (float, optional): The maximum number of seconds to wait for all requests to complete. If `None` (default), it waits indefinitely.

The `make_requests_with_retries` method includes one additional parameter:

*   `max_retries` (int, optional): The maximum number of times to retry a failed request. Defaults to 2.
