import pytest
from src.benchmark.data_models import SchemaManager

@pytest.fixture
def valid_record():
    return {
        "record_id": "recdya6FuYraBU5Rh",
        "question": "Astronomers are searching for exoplanets around two stars with exactly the same masses. Using the RV method, they detected one planet around each star, both with masses similar to Neptune. The stars have masses similar to our Sun and the planets are in circular orbits. Planet #1 showed a 5 miliangstrom periodic shift while Planet #2 showed 7 miliangstrom. How many times is the orbital period of planet #2 longer than that of planet #1?",
        "correct_answer": "~ 0.36",
        "correct_explanation": "Since orbits are circular and masses are same, RV amplitude K depends only on orbital period as K ~ P^(-1/3). K is proportional to wavelength shift, so comparing the shifts gives the period ratio.",
        "model_responses": {
            "deepseek_reasoner": {
                "answer": "0.36",
                "full_response": "Let's solve this step by step...",
                "reasoning": "Using Kepler's equations and RV method principles...",
                "grade": None
            }
        },
        "metadata": {
            "difficulty": "hard",
            "high_level_domain": "Astronomy",
            "subdomain": "Exoplanets"
        },
        "token_usage": {
            "deepseek_reasoner": {"input": 100, "output": 200}
        }
    }

def test_valid_record(valid_record):
    """Test that a valid record passes validation"""
    assert SchemaManager.validate_record(valid_record) is True

def test_invalid_record_type(valid_record):
    """Test that invalid field type fails validation"""
    invalid = valid_record.copy()
    invalid["record_id"] = 123  # Should be string
    assert SchemaManager.validate_record(invalid) is False

def test_missing_required_field(valid_record):
    """Test that missing required field fails validation"""
    invalid = valid_record.copy()
    del invalid["question"]
    assert SchemaManager.validate_record(invalid) is False

def test_nested_structure(valid_record):
    """Test that invalid nested field type fails validation"""
    invalid = valid_record.copy()
    invalid["metadata"]["difficulty"] = 123  # Should be string
    assert SchemaManager.validate_record(invalid) is False

def test_token_usage_structure(valid_record):
    """Test that invalid token usage type fails validation"""
    invalid = valid_record.copy()
    invalid["token_usage"]["deepseek_reasoner"]["input"] = "100"  # Should be int
    assert SchemaManager.validate_record(invalid) is False

def test_model_response_structure(valid_record):
    """Test that invalid model response type fails validation"""
    invalid = valid_record.copy()
    invalid["model_responses"]["deepseek_reasoner"]["answer"] = 42  # Should be string
    assert SchemaManager.validate_record(invalid) is False

def test_missing_metadata_field(valid_record):
    """Test that missing metadata field fails validation"""
    invalid = valid_record.copy()
    del invalid["metadata"]["difficulty"]
    assert SchemaManager.validate_record(invalid) is False

def test_invalid_metadata_structure(valid_record):
    """Test that invalid metadata structure fails validation"""
    invalid = valid_record.copy()
    invalid["metadata"] = []  # Should be dict
    assert SchemaManager.validate_record(invalid) is False

def test_invalid_token_usage_structure(valid_record):
    """Test that invalid token usage structure fails validation"""
    invalid = valid_record.copy()
    invalid["token_usage"]["deepseek_reasoner"] = []  # Should be dict
    assert SchemaManager.validate_record(invalid) is False

def test_invalid_model_responses_structure(valid_record):
    """Test that invalid model responses structure fails validation"""
    invalid = valid_record.copy()
    invalid["model_responses"] = []  # Should be dict
    assert SchemaManager.validate_record(invalid) is False

def test_none_grade_field(valid_record):
    """Test that None is valid for grade field"""
    assert SchemaManager.validate_record(valid_record) is True
    
    # Test with explicit None
    modified = valid_record.copy()
    modified["model_responses"]["deepseek_reasoner"]["grade"] = None
    assert SchemaManager.validate_record(modified) is True

def test_invalid_grade_field(valid_record):
    """Test that non-None value for grade field fails validation"""
    invalid = valid_record.copy()
    invalid["model_responses"]["deepseek_reasoner"]["grade"] = "A"  # Should be None
    assert SchemaManager.validate_record(invalid) is False