"""Module to expose version information."""
from importlib import metadata

__version__ = metadata.version("llm_api_client")
__version_info__ = tuple(__version__.split("."))
