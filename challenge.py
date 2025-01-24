# Standard library imports
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

# Third-party imports
import pandas as pd
from anthropic import Anthropic
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Configuration
class Config:
    # Check for .env file
    ENV_FILE = ".env"
    if not os.path.exists(ENV_FILE):
        raise FileNotFoundError(
            f"Missing {ENV_FILE} file. Please create one with DEEPSEEK_API_KEY, "
            "ANTHROPIC_API_KEY, and HF_TOKEN."
        )
    
    # Load environment variables
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    
    MAX_SAMPLES: Optional[int] = 1  # Set to None to run on all samples
    DATASET_NAME: str = "iDavidRein/gpqa"
    DATASET_SPLIT: str = "train"  # Only split available in GPQA
    RESULTS_DIR: str = "benchmark_results"
    
    # API configuration 
    DEEPSEEK_API_KEY: str = os.environ.get('DEEPSEEK_API_KEY')
    ANTHROPIC_API_KEY: str = os.environ.get('ANTHROPIC_API_KEY')
    HF_TOKEN: str = os.environ.get('HF_TOKEN')
    
    # Validate API keys
    if not all([DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN]):
        raise ValueError("Missing API keys. Please check your .env file.")
    
    # Model configuration
    DEEPSEEK_MODEL: str = "deepseek-reasoner"
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    
    # Retry configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2

# Add these constants after the Config class
MODEL_CONFIG = [
    {
        "model_key": "deepseek_reasoner",
        "display_name": "DeepSeek Reasoner",
        "token_groups": [{"type": "single", "input": "input", "output": "output"}]
    },
    {
        "model_key": "claude_sonnet",
        "display_name": "Claude Sonnet",
        "token_groups": [
            {"type": "grouped", "label": "Standalone", "input": "standalone_input", "output": "standalone_output"},
            {"type": "grouped", "label": "With Reasoning", "input": "with_reasoning_input", "output": "with_reasoning_output"}
        ]
    }
]

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

class BaseDataset(ABC):
    @abstractmethod
    def get_formatted_prompt(self, problem: dict) -> str:
        pass

class GPQADataset(BaseDataset):
    def get_formatted_prompt(self, problem: dict) -> str:
        """Format the prompt using the Question field directly"""
        return (
            f"{problem['Question']}\n\n"
            "Please provide a detailed explanation of your reasoning, "
            "then state your final answer clearly enclosed in <answer>...</answer> XML tags."
        )

def get_output_paths() -> Tuple[str, str]:
    """Generate standardized output paths for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"arc__deepseek-{Config.DEEPSEEK_MODEL}__claude-{Config.CLAUDE_MODEL}__{timestamp}"
    csv_path = os.path.join(Config.RESULTS_DIR, f"{base_name}.csv")
    json_path = os.path.join(Config.RESULTS_DIR, f"{base_name}.json")
    return csv_path, json_path

def extract_answer(text: str) -> Optional[str]:
    """Extract answer from between XML tags."""
    if not text:
        return None
    
    # Clean and normalize input text
    text = text.strip()
    if not text:
        return None
    
    # Check for malformed XML patterns
    if text == '</answer>' or text == '<answer>':
        return None
    
    pattern = r'<answer[^>]*>(.*?)</answer>'  # Modified to handle attributes
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not match:
        # If text has any XML-like tags but didn't match properly, return None
        if re.search(r'</?answer.*?>', text, re.IGNORECASE):
            return None
        # If no XML tags at all, return the full text
        return text.strip()
    
    # Extract and clean the answer text
    answer = match.group(1)
    # Remove any extra angle brackets at the start
    answer = re.sub(r'^>+', '', answer)
    # Normalize whitespace while preserving newlines
    lines = [line.strip() for line in answer.splitlines()]
    return '\n'.join(line for line in lines if line)

class ModelRunner:
    def __init__(self):
        self.deepseek_client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        self.anthropic_client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    def get_deepseek_response(self, prompt: str) -> Tuple[str, str]:
        """Get response from DeepSeek with retry logic."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.deepseek_client.chat.completions.create(
                    model=Config.DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                
                reasoning_content = ""
                content = ""
                
                for chunk in response:
                    if chunk.choices[0].delta.reasoning_content:
                        reasoning_content += chunk.choices[0].delta.reasoning_content
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                
                return reasoning_content, content
                
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise
                time.sleep(Config.RETRY_DELAY)

    def get_claude_response(self, prompt: str) -> str:
        """Get response from Claude with retry logic."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.anthropic_client.messages.create(
                    model=Config.CLAUDE_MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise
                time.sleep(Config.RETRY_DELAY)

def estimate_cost(num_samples: int) -> Dict[str, Any]:
    """Estimate API costs for running the benchmark using a data-driven approach."""
    # Load configuration
    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing {config_path} file")

    with open(config_path, 'r') as f:
        config = json.load(f)

    results = {
        "estimated_costs": {},
        "token_breakdown": {},
        "total_tokens": 0,
        "pricing_rates": {}
    }

    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for model in config["models"]:
        model_key = model["model_key"]
        tokens = model["tokens"]
        rates = model["cost_per_million"]

        model_cost = 0
        model_tokens = {"input": 0, "output": 0}

        for group in model["token_groups"]:
            input_tokens = tokens[group["input"]]
            output_tokens = tokens[group["output"]]
            
            # Calculate costs
            input_cost = (input_tokens * rates["input"] / 1_000_000) * num_samples
            output_cost = (output_tokens * rates["output"] / 1_000_000) * num_samples
            model_cost += input_cost + output_cost

            # Track tokens
            model_tokens["input"] += input_tokens * num_samples
            model_tokens["output"] += output_tokens * num_samples

        total_cost += model_cost
        total_input_tokens += model_tokens["input"]
        total_output_tokens += model_tokens["output"]
        
        results["estimated_costs"][model_key] = round(model_cost, 2)
        results["token_breakdown"][model_key] = model_tokens
        results["pricing_rates"][model_key] = rates

    results["estimated_costs"]["total"] = round(total_cost, 2)
    results["total_tokens"] = total_input_tokens + total_output_tokens
    results["total_input_tokens"] = total_input_tokens
    results["total_output_tokens"] = total_output_tokens

    return results

def print_cost_estimate(estimate: Dict[str, Any]) -> None:
    """Print cost estimate in a structured format"""
    # Load configuration to get model display info
    with open("config.json", 'r') as f:
        config = json.load(f)

    print("\n=== Cost Estimate ===")
    for model in config["models"]:
        model_key = model["model_key"]
        cost = estimate["estimated_costs"][model_key]
        print(f"{model['display_name']}: ${cost:.2f}")
    
    print(f"\nTotal Cost: ${estimate['estimated_costs']['total']:.2f}")
    
    print("\n=== Token Breakdown ===")
    for model in config["models"]:
        model_key = model["model_key"]
        tokens = estimate["token_breakdown"][model_key]
        
        print(f"\n{model['display_name']}:")
        for group in model["token_groups"]:
            if group["type"] == "grouped":
                print(f"  {group['label']}:")
            print(f"    Input: {tokens['input']:,}")
            print(f"    Output: {tokens['output']:,}")

    print("\n=== Totals ===")
    print(f"Total Input Tokens: {estimate['total_input_tokens']:,}")
    print(f"Total Output Tokens: {estimate['total_output_tokens']:,}")
    print(f"Total Tokens: {estimate['total_tokens']:,}")

def run_benchmark():
    """Main benchmark function."""
    setup_directories()
    csv_path, json_path = get_output_paths()
    
    # Load dataset
    dataset = load_dataset(Config.DATASET_NAME, "gpqa_diamond", token=Config.HF_TOKEN)
    data = dataset[Config.DATASET_SPLIT]
    total_samples = len(data)
    
    # If MAX_SAMPLES is set, take a random sample
    if Config.MAX_SAMPLES:
        random.seed(99)  # For reproducibility
        data = random.sample(list(data), Config.MAX_SAMPLES)
        num_samples = Config.MAX_SAMPLES
    else:
        num_samples = total_samples
    
    # Show cost estimate
    cost_estimate = estimate_cost(num_samples)
    print_cost_estimate(cost_estimate)

    # Ask for confirmation
    response = input("\nDo you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborting benchmark.")
        return

    gpqa_dataset = GPQADataset()
    # Initialize model runner
    runner = ModelRunner()
    
    # Initialize results DataFrame with new columns
    results_df = pd.DataFrame(columns=[
        "record_id", "question", "correct_answer",
        "r1_full_response", "r1_answer",
        "claude_full_response", "claude_answer",
        "claude_with_r1_full_response", "claude_with_r1_answer",
        "subdomain", "high_level_domain", "difficulty"
    ])
    results_df.to_csv(csv_path, index=False)
    
    # Initialize metrics
    metrics = {
        "total_samples_processed": 0,
        "errors": 0,
        "domains": {}
    }
    
    # Main evaluation loop
    for idx, example in tqdm(enumerate(data), total=len(data)):
        try:
            # Format prompt
            prompt = gpqa_dataset.get_formatted_prompt(example)
            
            # Get model responses
            reasoning, r1_response = runner.get_deepseek_response(prompt)
            claude_response = runner.get_claude_response(prompt)
            claude_r1_response = runner.get_claude_response(
                f"{prompt}\n\n<reasoning>{reasoning}</reasoning>"
            )
            
            # Extract answers from XML tags
            r1_answer = extract_answer(r1_response)
            claude_answer = extract_answer(claude_response)
            claude_r1_answer = extract_answer(claude_r1_response)
            
            # Store result
            result = {
                "record_id": example['Record ID'],
                "question": example['Question'],
                "correct_answer": example['Correct Answer'], 
                "r1_full_response": r1_response,
                "r1_answer": r1_answer,
                "claude_full_response": claude_response,
                "claude_answer": claude_answer,
                "claude_with_r1_full_response": claude_r1_response,
                "claude_with_r1_answer": claude_r1_answer,
                "subdomain": example['Subdomain'],
                "high_level_domain": example['High-level domain'],
                "difficulty": example["Writer's Difficulty Estimate"]
            }
            # Update domain statistics
            domain = example['High-level domain']
            if domain not in metrics["domains"]:
                metrics["domains"][domain] = 0
            metrics["domains"][domain] += 1
            
            # Append to CSV immediately
            pd.DataFrame([result]).to_csv(csv_path, mode='a', header=False, index=False)
            
            metrics["total_samples_processed"] += 1
            
        except Exception as e:
            metrics["errors"] += 1
            print(f"\nError processing example {example['Record ID']}: {str(e)}")
            continue
        
        # Update metrics file
        metrics["timestamp"] = datetime.now().isoformat()
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print(f"\nBenchmark completed!")
    print(f"Results saved to: {csv_path}")
    print(f"Metrics saved to: {json_path}")
    print(f"\nProcessed {metrics['total_samples_processed']} samples")
    print(f"Errors: {metrics['errors']}")
    print("\nSamples by domain:")
    for domain, count in metrics["domains"].items():
        print(f"{domain}: {count}")

if __name__ == "__main__":
    run_benchmark()