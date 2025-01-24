import pytest
from challenge import validate_record, SchemaManager

@pytest.fixture
def valid_record():
    return {
        "record_id": "test123",
        "question": "What is physics?",
        "correct_answer": "The study of nature",
        "correct_explanation": "Study of matter and energy",
        "model_responses": {
            "deepseek_reasoner": {
                "answer": "42",
                "full_response": "Long response...",
                "reasoning": "Deep thoughts...",
                "grade": None
            }
        },
        "metadata": {
            "difficulty": "hard",
            "high_level_domain": "Physics",
            "subdomain": "Basics"
        },
        "token_usage": {
            "deepseek_reasoner": {"input": 100, "output": 200}
        }
    }

def test_valid_record(valid_record):
    assert validate_record(valid_record) is True

def test_invalid_record_type(valid_record):
    invalid = valid_record.copy()
    invalid["record_id"] = 123  # Should be string
    assert validate_record(invalid) is False

def test_missing_required_field(valid_record):
    invalid = valid_record.copy()
    del invalid["question"]
    assert validate_record(invalid) is False

def test_nested_structure(valid_record):
    invalid = valid_record.copy()
    invalid["metadata"]["difficulty"] = 123  # Should be string
    assert validate_record(invalid) is False

def test_token_usage_structure(valid_record):
    invalid = valid_record.copy()
    invalid["token_usage"]["deepseek_reasoner"]["input"] = "100"  # Should be int
    assert validate_record(invalid) is False

def test_model_response_structure(valid_record):
    invalid = valid_record.copy()
    invalid["model_responses"]["deepseek_reasoner"]["answer"] = 42  # Should be string
    assert validate_record(invalid) is False