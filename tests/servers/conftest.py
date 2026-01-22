"""Shared fixtures for server tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def preserve_working_directory():
    """Automatically preserve and restore working directory for all tests."""
    original_cwd = os.getcwd()
    yield
    # Restore working directory after each test
    try:
        os.chdir(original_cwd)
    except OSError:
        # Directory might have been deleted, that's ok
        pass
