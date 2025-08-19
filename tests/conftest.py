"""Fixtures for pytest tests in the project.
"""
import pytest


@pytest.fixture(params=[
    # Test case 1: Long message at the end
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "a tree " * 10_000}  # Very long message
    ],
    # Test case 2: Long message in the middle
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "a tree " * 10_000},  # Very long message
        {"role": "assistant", "content": "I'll help you with trees."}
    ],
    # Test case 3: Multiple messages of similar length
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "A " * 5000},
        {"role": "assistant", "content": "B " * 5000},
        {"role": "user", "content": "C " * 5000}
    ],
    # Test case 4: Mixed message lengths with long system prompt
    [
        {"role": "system", "content": "You are a helpful assistant. " * 5000},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Tell me about trees. " * 500}
    ],
    # Test case 5: Short content but many messages
    [{"role": ("user" if i % 2 == 0 else "assistant"), "content": f"Message {i}" * 100}
     for i in range(20)]
], ids=["long_end", "long_middle", "multiple_similar", "long_system", "many_messages"])
def test_messages(request):
    """Fixture providing various message configurations for testing.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest request object

    Returns
    -------
    list
        A list of message dictionaries with different configurations
    """
    return request.param
