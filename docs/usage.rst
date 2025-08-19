Usage Examples
==============

Basic concurrent requests
-------------------------

.. code-block:: python

   from llm_api_client import APIClient

   client = APIClient(max_requests_per_minute=200, max_tokens_per_minute=200000)

   prompts = [
       "Summarize the plot of The Matrix in one paragraph.",
       "List three benefits of unit testing.",
       "Translate 'good morning' to Spanish.",
   ]

   requests_data = [
       {
           "model": "gpt-4o-mini",
           "messages": [{"role": "user", "content": prompt}],
           "temperature": 0.4,
       }
       for prompt in prompts
   ]

   responses = client.make_requests(requests_data)

   for r in responses:
       print(r.choices[0].message.content)

Retries with backoff
--------------------

Use :py:meth:`llm_api_client.api_client.APIClient.make_requests_with_retries` to automatically retry failed calls:

.. code-block:: python

   from llm_api_client import APIClient

   client = APIClient(max_requests_per_minute=100)

   requests_data = [
       {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]}
       for _ in range(10)
   ]

   responses = client.make_requests_with_retries(
       requests_data,
       max_retries=2,
       sanitize=True,
       timeout=60,
   )

   print(client.tracker)  # usage stats

Streaming and large context handling
------------------------------------

Requests are sanitized to fit model/provider constraints when ``sanitize=True``.

.. code-block:: python

   from llm_api_client import APIClient

   client = APIClient()

   long_history = [
       {"role": "user", "content": "... a very long conversation ..."},
       # many messages
   ]

   response = client.make_requests([
       {"model": "gpt-4o-mini", "messages": long_history}
   ])[0]

Accessing history and usage
---------------------------

.. code-block:: python

   print(client.history)            # list of request/response entries
   print(client.tracker.details)    # dict with costs/tokens/latency stats

