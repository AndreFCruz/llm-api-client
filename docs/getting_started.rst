Getting Started
===============

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install llm-api-client

Quick start
-----------

Create a simple client and make concurrent requests:

.. code-block:: python

   from llm_api_client import APIClient

   client = APIClient(
       max_requests_per_minute=1000,
       max_tokens_per_minute=100000,
   )

   prompts = [
       "Explain the theory of relativity in simple terms.",
       "Write a short poem about a cat.",
       "What is the capital of France?",
   ]

   requests_data = [
       {
           "model": "gpt-3.5-turbo",
           "messages": [{"role": "user", "content": prompt}],
       }
       for prompt in prompts
   ]

   responses = client.make_requests(requests_data)

   for i, response in enumerate(responses):
       if response:
           try:
               print(response.choices[0].message.content)
           except Exception:
               print(response)
       else:
           print(f"Request {i+1} failed")

