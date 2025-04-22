# `llm-api-client` :robot::zap:

![Tests status](https://github.com/AndreFCruz/llm-api-client/actions/workflows/tests.yml/badge.svg)
![Docs status](https://github.com/AndreFCruz/llm-api-client/actions/workflows/pypi-publish.yml/badge.svg)
![PyPI status](https://github.com/AndreFCruz/llm-api-client/actions/workflows/pypi-publish.yml/badge.svg)
![PyPI version](https://badgen.net/pypi/v/llm-api-client)
![PyPI - License](https://img.shields.io/pypi/l/llm-api-client)
![Python compatibility](https://badgen.net/pypi/python/llm-api-client)

A Python helper library for efficiently managing concurrent, rate-limited API requests, especially for Large Language Models (LLMs) via [LiteLLM](https://github.com/BerriAI/litellm).

It provides an `APIClient` that handles:
*   **Concurrency:** Making multiple API calls simultaneously using threads.
*   **Rate Limiting:** Respecting API limits for requests per minute (RPM) and tokens per minute (TPM).
*   **Retries:** Automatically retrying failed requests.
*   **Request Sanitization:** Cleaning up request parameters to ensure compatibility with different models/providers.
*   **Context Management:** Truncating message history to fit within model context windows.
*   **Usage Tracking:** Monitoring API costs, token counts, and response times via an integrated `APIUsageTracker`.

## Installation

Install the package directly from PyPI:

```bash
pip install llm-api-client
```

## Usage

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
