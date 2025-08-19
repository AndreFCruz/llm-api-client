"""Tests for `_params.py` constants.

Ensure the exported lists are non-empty and contain expected items, and that
`ALL_COMPLETION_PARAMS` is a union that includes unique entries.
"""
from llm_api_client._params import (
    OPENAI_COMPLETION_PARAMS,
    LITELLM_COMPLETION_PARAMS,
    ALL_COMPLETION_PARAMS,
)


def test_params_lists_non_empty():
    assert isinstance(OPENAI_COMPLETION_PARAMS, list) and OPENAI_COMPLETION_PARAMS
    assert isinstance(LITELLM_COMPLETION_PARAMS, list) and LITELLM_COMPLETION_PARAMS
    assert isinstance(ALL_COMPLETION_PARAMS, list) and ALL_COMPLETION_PARAMS


def test_common_expected_params_present():
    expected_openai = {"temperature", "max_tokens", "top_p", "tools"}
    expected_litellm = {"num_retries", "request_timeout", "api_base"}

    assert expected_openai.issubset(set(OPENAI_COMPLETION_PARAMS))
    assert expected_litellm.issubset(set(LITELLM_COMPLETION_PARAMS))


def test_all_params_contains_both_sets_without_missing_items():
    openai_set = set(OPENAI_COMPLETION_PARAMS)
    litellm_set = set(LITELLM_COMPLETION_PARAMS)
    all_set = set(ALL_COMPLETION_PARAMS)

    assert openai_set.issubset(all_set)
    assert litellm_set.issubset(all_set)

