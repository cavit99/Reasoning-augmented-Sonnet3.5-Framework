import pytest

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment"""
    # No setup needed - file operations are mocked in costtest.py
    pass