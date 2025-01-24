import sys
from pathlib import Path
import json

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from challenge import extract_answer, GPQADataset


def test_extract_answer():
    """Test answer extraction from XML tags with various formats."""
    test_cases = [
        # Basic format
        (
            "Let me explain... <answer>The key issue is data interpretation errors.</answer>",
            "The key issue is data interpretation errors."
        ),
        
        # Multi-line answer
        (
            """Here's my reasoning...
            <answer>
            The main factors are:
            1. Data interpretation
            2. Analysis errors
            </answer>""",
            "The main factors are:\n1. Data interpretation\n2. Analysis errors"
        ),
        
        # Case insensitivity
        (
            "Analysis shows... <ANSWER>Configuration errors</ANSWER>",
            "Configuration errors"
        ),
        
        # No tags - should return full text
        (
            "The answer is configuration errors",
            "The answer is configuration errors"
        ),
        
        # Empty tags
        (
            "<answer></answer>",
            ""
        ),
        
        # Multiple tags - should take first
        (
            "<answer>First answer</answer> <answer>Second answer</answer>",
            "First answer"
        ),
        
        # Edge cases
        (None, None),
        ("", None),
        ("<answer>", None),
        ("</answer>", None),
    ]
    
    for input_text, expected in test_cases:
        result = extract_answer(input_text)
        assert result == expected, f"\nInput: '{input_text}'\nExpected: {expected}\nGot: {result}"

def test_gpqa_prompt_formatting():
    """Test GPQA dataset prompt formatting."""
    dataset = GPQADataset()
    
    # Test basic prompt formatting
    problem = {
        "Question": "What is a common source of errors in bioinformatics analysis?",
        "Answer": "Data interpretation errors",
        "Explanation": "Data interpretation errors are common due to...",
    }
    
    prompt = dataset.get_formatted_prompt(problem)
    
    # Check prompt structure
    assert problem["Question"] in prompt
    assert "Please provide a detailed explanation" in prompt
    assert "<answer>" in prompt
    assert "</answer>" in prompt

def test_gpqa_prompt_special_characters():
    """Test GPQA prompt formatting with special characters."""
    dataset = GPQADataset()
    
    # Test with special characters and formatting
    problem = {
        "Question": "Why do these errors occur?\n1. Format issues\n2. Analysis errors",
        "Answer": "Multiple factors contribute...",
        "Explanation": "The explanation includes:\n- Point 1\n- Point 2",
    }
    
    prompt = dataset.get_formatted_prompt(problem)
    assert problem["Question"] in prompt
    assert prompt.count("\n") >= 2  # Should preserve newlines

def test_answer_extraction_with_reasoning():
    """Test extraction of answers when combined with reasoning."""
    test_cases = [
        # Reasoning followed by answer
        (
            """Let me analyze this step by step:
            1. First point
            2. Second point
            Therefore, <answer>The main cause is misconfiguration</answer>""",
            "The main cause is misconfiguration"
        ),
        
        # Answer with explanation
        (
            """<answer>Data format inconsistencies</answer>
            This is because...""",
            "Data format inconsistencies"
        ),
        
        # Complex formatting
        (
            """Analysis:
            * Point 1
            * Point 2
            
            <answer>
            Multiple factors:
            1. Data issues
            2. Process errors
            </answer>
            
            Additional notes...""",
            "Multiple factors:\n1. Data issues\n2. Process errors"
        ),
    ]
    
    for input_text, expected in test_cases:
        result = extract_answer(input_text)
        assert result == expected, f"\nInput: '{input_text}'\nExpected: {expected}\nGot: {result}"

def test_malformed_xml():
    """Test handling of malformed XML tags."""
    test_cases = [
        ("<answer>Incomplete tag", None),
        ("No closing bracket<answer>text</answer", None),
        ("<answer>Nested <answer>tags</answer></answer>", "Nested <answer>tags"),
        ("<answer attr='value'>With attributes</answer>", "With attributes"),
        ("<Answer>Case sensitive test</Answer>", "Case sensitive test"),  # Should work due to re.IGNORECASE
    ]
    
    for input_text, expected in test_cases:
        result = extract_answer(input_text)
        assert result == expected, f"\nInput: '{input_text}'\nExpected: {expected}\nGot: {result}"

def test_whitespace_handling():
    """Test handling of whitespace in answers."""
    test_cases = [
        (
            "<answer>\n    Indented answer\n</answer>",
            "Indented answer"
        ),
        (
            "<answer>   Extra spaces   </answer>",
            "Extra spaces"
        ),
        (
            "<answer>\nMultiple\nLines\n</answer>",
            "Multiple\nLines"
        ),
    ]
    
    for input_text, expected in test_cases:
        result = extract_answer(input_text)
        assert result == expected, f"\nInput: '{input_text}'\nExpected: {expected}\nGot: {result}"

def test_config_loading():
    """Test configuration loading and validation."""
    from challenge import Config
    
    assert hasattr(Config, 'DATASET_NAME')
    assert Config.DATASET_NAME == "iDavidRein/gpqa"
    assert Config.DATASET_SPLIT == "train"
    assert isinstance(Config.MAX_SAMPLES, (int, type(None)))

def test_model_runner_initialization():
    """Test ModelRunner initialization and client setup."""
    from challenge import ModelRunner
    
    runner = ModelRunner()
    assert hasattr(runner, 'deepseek_client')
    assert hasattr(runner, 'anthropic_client')

def test_output_paths():
    """Test output path generation."""
    from challenge import get_output_paths
    
    csv_path, json_path = get_output_paths()
    assert csv_path.endswith('.csv')
    assert json_path.endswith('.json')
    assert 'deepseek' in csv_path
    assert 'claude' in csv_path

def test_gpqa_dataset_loading():
    """Test GPQA dataset loading and structure."""
    from datasets import load_dataset
    from challenge import Config
    
    dataset = load_dataset(Config.DATASET_NAME, "gpqa_diamond", token=Config.HF_TOKEN)
    data = dataset[Config.DATASET_SPLIT]
    
    # Test required fields are present
    example = data[0]
    required_fields = [
        'Record ID', 'Question', 'Correct Answer', 'Explanation',
        'Subdomain', 'High-level domain', "Writer's Difficulty Estimate"
    ]
    for field in required_fields:
        assert field in example, f"Missing required field: {field}"

def test_combined_prompt_formatting():
    """Test prompt formatting with reasoning included."""
    from challenge import GPQADataset
    
    dataset = GPQADataset()
    reasoning = "This is the reasoning from the first model..."
    
    # Test basic prompt
    problem = {
        "Question": "Test question?",
        "Answer": "Test answer",
        "Explanation": "Test explanation"
    }
    
    basic_prompt = dataset.get_formatted_prompt(problem)
    assert "Test question?" in basic_prompt
    assert "<answer>" in basic_prompt
    
    # Test combined prompt with reasoning
    combined_prompt = f"{basic_prompt}\n\n<reasoning>{reasoning}</reasoning>"
    assert reasoning in combined_prompt
    assert combined_prompt.count("<answer>") == 1  # Should only have one set of answer tags

def test_metrics_structure():
    """Test metrics dictionary structure."""
    from challenge import run_benchmark
    
    # Initialize empty metrics
    metrics = {
        "total_samples_processed": 0,
        "errors": 0,
        "domains": {},
        "timestamp": None
    }
    
    assert "total_samples_processed" in metrics
    assert "errors" in metrics
    assert "domains" in metrics
    assert isinstance(metrics["domains"], dict)

def test_cost_estimation():
    """Test cost estimation calculation."""
    from challenge import estimate_cost
    
    # Test with sample size of 1
    cost_estimate = estimate_cost(1)
    assert "estimated_costs" in cost_estimate
    assert "token_estimates" in cost_estimate
    assert "pricing_info" in cost_estimate
    assert cost_estimate["num_samples"] == 1
    assert isinstance(cost_estimate["estimated_costs"]["total"], float)

def test_error_handling():
    """Test error handling in answer extraction."""
    test_cases = [
        # Malformed XML
        ("<answer>test", None),
        ("answer>test</answer>", None),  
        ("<answer>>test</answer>", "test"),  
        ("<answer>test</answer", None),
        
        # Empty or whitespace
        ("   ", None),
        ("\n\n", None),
        
        # Mixed case tags
        ("<ANSWER>Test</answer>", "Test"),
        ("<Answer>Test</ANSWER>", "Test"),
        
        # Nested tags handling
        ("<answer>outer <answer>inner</answer></answer>", "outer <answer>inner"),
    ]
    
    for input_text, expected in test_cases:
        result = extract_answer(input_text)
        assert result == expected, f"\nInput: '{input_text}'\nExpected: {expected}\nGot: {result}"

def test_pricing_config():
    """Test pricing configuration structure and values."""
    # Load pricing configuration
    pricing_config_path = Path(__file__).parent.parent / "pricing_config.json"
    assert pricing_config_path.exists(), "pricing_config.json not found"
    
    with open(pricing_config_path, 'r') as f:
        pricing_config = json.load(f)
    
    # Test structure
    assert "deepseek_reasoner" in pricing_config
    assert "claude_sonnet" in pricing_config
    
    # Test DeepSeek Reasoner config
    deepseek = pricing_config["deepseek_reasoner"]
    assert "tokens" in deepseek
    assert "cost_per_million" in deepseek
    assert all(k in deepseek["tokens"] for k in ["input", "output"])
    assert all(k in deepseek["cost_per_million"] for k in ["input", "output"])
    
    # Test Claude Sonnet config
    claude = pricing_config["claude_sonnet"]
    assert "tokens" in claude
    assert "cost_per_million" in claude
    assert all(k in claude["tokens"] for k in [
        "standalone_input", "standalone_output",
        "with_reasoning_input", "with_reasoning_output"
    ])
    assert all(k in claude["cost_per_million"] for k in ["input", "output"])
    
    # Test value types and ranges
    def validate_numeric(value, name):
        assert isinstance(value, (int, float)), f"{name} should be numeric"
        assert value >= 0, f"{name} should be non-negative"
    
    # Validate DeepSeek values
    for k, v in deepseek["tokens"].items():
        validate_numeric(v, f"deepseek_reasoner.tokens.{k}")
    for k, v in deepseek["cost_per_million"].items():
        validate_numeric(v, f"deepseek_reasoner.cost_per_million.{k}")
    
    # Validate Claude values
    for k, v in claude["tokens"].items():
        validate_numeric(v, f"claude_sonnet.tokens.{k}")
    for k, v in claude["cost_per_million"].items():
        validate_numeric(v, f"claude_sonnet.cost_per_million.{k}")
    
    # Test specific values
    assert deepseek["tokens"]["input"] == 108
    assert deepseek["tokens"]["output"] == 515
    assert deepseek["cost_per_million"]["input"] == 0.14
    assert deepseek["cost_per_million"]["output"] == 2.19
    
    assert claude["tokens"]["standalone_input"] == 122
    assert claude["tokens"]["standalone_output"] == 260
    assert claude["tokens"]["with_reasoning_input"] == 545
    assert claude["tokens"]["with_reasoning_output"] == 239
    assert claude["cost_per_million"]["input"] == 3.00
    assert claude["cost_per_million"]["output"] == 15.00

def test_ai_response_patterns():
    """Test common AI response patterns and edge cases."""
    test_cases = [
        # Common AI response patterns
        (
            "Let me think about this step by step:\n1. First...\n2. Second...\n<answer>Final conclusion</answer>",
            "Final conclusion"
        ),
        
        # Multiple reasoning attempts
        (
            "First attempt... <answer>Answer 1</answer>\nOn second thought... <answer>Answer 2</answer>",
            "Answer 1"  # Should take first answer only
        ),
        
        # Common formatting mistakes by AI
        (
            "<Answer>Here's why:\n1. Point one\n2. Point two</Answer>",
            "Here's why:\n1. Point one\n2. Point two"
        ),
        
        # XML-like formatting in the answer itself
        (
            "<answer>The error occurs in <filename>data.txt</filename> when processing</answer>",
            "The error occurs in <filename>data.txt</filename> when processing"
        ),
        
        # Reasoning tags mixed with answer tags
        (
            "<reasoning>Analysis...</reasoning><answer>Conclusion</answer>",
            "Conclusion"
        ),
        
        # Common markdown/formatting in answers
        (
            "<answer>**Bold text**\n- Bullet point\n```code block```</answer>",
            "**Bold text**\n- Bullet point\n```code block```"
        ),
        
        # Unicode and special characters
        (
            "<answer>π ≈ 3.14159\n→ Therefore...</answer>",
            "π ≈ 3.14159\n→ Therefore..."
        ),
        
        # HTML-like formatting
        (
            "<answer><b>Important:</b> The key finding is...</answer>",
            "<b>Important:</b> The key finding is..."
        ),
    ]
    
    for input_text, expected in test_cases:
        result = extract_answer(input_text)
        assert result == expected, f"\nInput: '{input_text}'\nExpected: {expected}\nGot: {result}" 