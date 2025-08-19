Configuration
=============

Client limits
-------------

- **max_requests_per_minute**: RPM limiter for concurrency and scheduling.
- **max_tokens_per_minute**: TPM limiter for budgeted token throughput.
- **max_workers**: Upper bound for thread pool size (defaults to ``min(RPM, CPU*20)``).

Environment variables
---------------------

- **DEFAULT_MAX_CONTEXT_TOKENS**: Maximum context window tokens used by the sanitizer
  to truncate long message histories (default: 20000).

Provider credentials
--------------------

Set the credentials expected by your LiteLLM provider. For OpenAI-compatible APIs,
set ``OPENAI_API_KEY``.

Logging
-------

The library uses the standard ``logging`` module. Configure at application startup:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

Autodoc and docs
----------------

The Sphinx configuration mocks heavy dependencies (``openai``, ``pyrate_limiter``, ``numpy``)
to avoid import issues during doc builds.

