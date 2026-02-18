"""Shared test fixtures for ANVIL."""

from __future__ import annotations

import pytest

from anvil.config import AnvilConfig


@pytest.fixture
def sample_config():
    """Return a default AnvilConfig for testing."""
    return AnvilConfig()
