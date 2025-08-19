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

Full documentation: [https://andrefcruz.github.io/llm-api-client/](https://andrefcruz.github.io/llm-api-client/)

For API reference and more examples, see:
- Getting Started: [https://andrefcruz.github.io/llm-api-client/getting_started.html](https://andrefcruz.github.io/llm-api-client/getting_started.html)
- Usage Guide: [https://andrefcruz.github.io/llm-api-client/usage.html](https://andrefcruz.github.io/llm-api-client/usage.html)
- Configuration: [https://andrefcruz.github.io/llm-api-client/configuration.html](https://andrefcruz.github.io/llm-api-client/configuration.html)
- API Reference: [https://andrefcruz.github.io/llm-api-client/api.html](https://andrefcruz.github.io/llm-api-client/api.html)

## Installation

Install the package directly from PyPI:

```bash
pip install llm-api-client
```

## Quick start

### Single request

```python
from llm_api_client import APIClient

client = APIClient()  # Defaults approximate OpenAI Tier 4 limits

responses = client.make_requests([
    {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    }
])

print(responses[0].choices[0].message.content)
```

### With retries

```python
responses = client.make_requests_with_retries([
    {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi!"}]}
], max_retries=2)
```

### Disable request sanitization (advanced)

```python
responses = client.make_requests([
    {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi!"}]}
], sanitize=False)
```

### Control concurrency and rate limits

```python
client = APIClient(max_requests_per_minute=600, max_tokens_per_minute=100_000, max_workers=50)
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
        "model": "gpt-4o-mini",
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
```

### Client Parameters

The `APIClient` constructor accepts:

- `max_requests_per_minute` (int, default `10000`): Maximum API requests per minute (RPM).
- `max_tokens_per_minute` (int, default `2000000`): Maximum tokens per minute (TPM).
- `max_workers` (int, optional): Maximum worker threads. If not set, defaults to `max_requests_per_minute` when provided, otherwise to `CPU count * 20`.

### Method Parameters

Both `make_requests` and `make_requests_with_retries` accept the following core parameters:

*   `requests` (list[dict]): A list where each dictionary represents the parameters for a single API call (e.g., `model`, `messages`, `temperature`, etc.) -- follows the openai API standard via [`litellm`](https://github.com/BerriAI/litellm).
*   `max_workers` (int, optional): Maximum concurrent threads. Defaults to the client's configured worker count (set via the constructor).
*   `sanitize` (bool, optional): If `True` (default), the client will attempt to remove parameters that are incompatible with the specified model and provider before making the request. It also truncates message history to fit the model's context window.
*   `timeout` (float, optional): The maximum number of seconds to wait for all requests to complete. If `None` (default), it waits indefinitely.

The `make_requests_with_retries` method includes one additional parameter:

*   `max_retries` (int, optional): The maximum number of times to retry a failed request. Defaults to 2.