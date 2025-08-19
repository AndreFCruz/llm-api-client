"""Module to expose version information.

Resilient to running from source without an installed distribution.
"""
from importlib import metadata

try:
    # Project name in pyproject is "llm-api-client"
    __version__ = metadata.version("llm-api-client")
except metadata.PackageNotFoundError:
    # Fallback when running from source without installation
    __version__ = "0.1.2"

__version_info__ = tuple(__version__.split("."))
