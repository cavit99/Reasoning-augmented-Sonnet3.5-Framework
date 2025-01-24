import os
import shutil
import pytest

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment with mock config.json"""
    # Create a temporary directory for test results
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Copy test config.json to root directory
    test_config = os.path.join(os.path.dirname(__file__), "config.json")
    shutil.copy(test_config, "config.json")
    
    # Clean up after tests
    yield
    
    # Remove temporary files
    if os.path.exists("config.json"):
        os.remove("config.json")
    if os.path.exists("benchmark_results"):
        shutil.rmtree("benchmark_results") 