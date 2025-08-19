"""Smoke tests for public package exports and version module behavior."""

import llm_api_client as pkg


def test_public_exports_exist():
    assert hasattr(pkg, "APIClient")
    assert hasattr(pkg, "APIUsageTracker")
    assert isinstance(pkg.APIClient, type)
    assert isinstance(pkg.APIUsageTracker, type)


def test_version_metadata_available():
    # __version__ should be a non-empty string like "0.x.y"
    assert isinstance(pkg.__version__, str) and pkg.__version__
    # __version_info__ should be a tuple of components
    assert isinstance(pkg.__version_info__, tuple)
    assert len(pkg.__version_info__) >= 2
