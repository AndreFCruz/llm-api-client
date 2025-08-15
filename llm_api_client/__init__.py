"""A client for interacting with LLM completion APIs and tracking usage."""
from ._version import __version__, __version_info__
from .api_client import APIClient
from .api_tracker import APIUsageTracker

__all__ = ["APIClient", "APIUsageTracker"]
