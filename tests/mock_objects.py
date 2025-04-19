"""Mock objects for testing.

This module contains mock implementations and other test utilities that can be
shared across multiple test files.
"""
import time
from unittest.mock import MagicMock


class MockResponse:
    """Mock response object that mimics LiteLLM API responses.

    Parameters
    ----------
    content : str
        Content to return in the response
    model : str
        Model name used for the response
    """
    def __init__(self, content="Test response", model="gpt-3.5-turbo"):
        self.choices = [
            MagicMock(
                message=MagicMock(
                    content=content,
                    role="assistant"
                )
            )
        ]
        self.model = model
        self.created = int(time.time())
        self.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }

    def model_dump(self):
        """
        Returns
        -------
        dict
            Dictionary representation of the response
        """
        return {
            "choices": [{"message": {"content": self.choices[0].message.content, "role": "assistant"}}],
            "model": self.model,
            "created": self.created,
            "usage": self.usage
        }

    def __getitem__(self, key):
        """Make the object subscriptable for accessing attributes with brackets.

        Parameters
        ----------
        key : str
            The key to access

        Returns
        -------
        Any
            The value at the specified key
        """
        return getattr(self, key)
