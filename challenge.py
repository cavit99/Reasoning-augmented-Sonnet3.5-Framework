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
from datasets import load_dataset, Dataset, concatenate_datasets
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
    
    # Add HF_DATASET_REPO to Config class
    HF_DATASET_REPO: str = "spawn99/GPQA-diamond-ClaudeR1"

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

def get_output_paths() -> str:
    """Generate standardized output path for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"arc__deepseek-{Config.DEEPSEEK_MODEL}__claude-{Config.CLAUDE_MODEL}__{timestamp}"
    jsonl_path = os.path.join(Config.RESULTS_DIR, f"{base_name}.jsonl")
    return jsonl_path

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

def get_processed_record_ids(repo_id: str) -> set[str]:
    """Get set of already processed record IDs efficiently."""
    try:
        # Stream just the record_id column to handle potentially large datasets
        dataset = load_dataset(
            repo_id,
            split="train",
            streaming=True,
            columns=["record_id"]
        )
        
        processed_ids = set()
        for batch in dataset.iter(batch_size=1000):
            processed_ids.update(batch["record_id"])
            
        print(f"Loaded {len(processed_ids)} record IDs from existing dataset")
        return processed_ids
        
    except Exception as e:
        print(f"No existing dataset found or error occurred: {e}")
        return set()

def run_benchmark():
    """Main benchmark function."""
    setup_directories()
    
    # Add local backup path
    jsonl_path = get_output_paths()
    
    try:
        processed_ids = get_processed_record_ids(Config.HF_DATASET_REPO)
    except Exception as e:
        print(f"Error checking existing dataset: {e}")
        return
    
    # Load source dataset
    dataset = load_dataset(Config.DATASET_NAME, "gpqa_diamond", token=Config.HF_TOKEN)
    data = dataset[Config.DATASET_SPLIT]
    total_samples = len(data)
    print(f"Total samples in dataset: {total_samples}")
    
    # Get unprocessed records first
    unprocessed_data = [d for d in data if d['Record ID'] not in processed_ids]
    print(f"Unprocessed samples available: {len(unprocessed_data)}")
    
    # If MAX_SAMPLES is set, take a random sample from UNPROCESSED data
    if Config.MAX_SAMPLES:
        random.seed(99)  # For reproducibility
        if len(unprocessed_data) <= Config.MAX_SAMPLES:
            print(f"Only {len(unprocessed_data)} unprocessed samples available")
            num_samples = len(unprocessed_data)
            data = unprocessed_data
        else:
            num_samples = Config.MAX_SAMPLES
            data = random.sample(unprocessed_data, Config.MAX_SAMPLES)
    else:
        num_samples = len(unprocessed_data)
        data = unprocessed_data
    
    if num_samples == 0:
        print("All samples have been processed already!")
        return
        
    print(f"Will process {num_samples} samples")
    
    # Show cost estimate for samples to process
    cost_estimate = estimate_cost(num_samples)
    print_cost_estimate(cost_estimate)

    # Ask for confirmation
    response = input("\nDo you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborting benchmark.")
        return

    gpqa_dataset = GPQADataset()
    runner = ModelRunner()
    
    # Initialize metrics
    metrics = {
        "total_samples_processed": len(processed_ids),
        "errors": 0,
        "domains": {},
        "cost_estimate": cost_estimate,
        "timestamp": datetime.now().isoformat()
    }
    
    success = True  # Track if all processing completed successfully
    
    # Main evaluation loop
    for question_data in tqdm(data, total=len(data)):
        record_id = question_data['Record ID']
        
        if record_id in processed_ids:
            tqdm.write(f"\nSkipping already processed record: {record_id}")
            continue
            
        try:
            prompt = gpqa_dataset.get_formatted_prompt(question_data)
            
            # Get model responses
            reasoning, r1_response = runner.get_deepseek_response(prompt)
            claude_response = runner.get_claude_response(prompt)
            claude_r1_response = runner.get_claude_response(
                f"{prompt}\n\n<reasoning>{reasoning}</reasoning>"
            )
            
            # Extract answers
            r1_answer = extract_answer(r1_response)
            claude_answer = extract_answer(claude_response)
            claude_r1_answer = extract_answer(claude_r1_response)
            
            # Create result dictionary
            result = {
                "record_id": record_id,
                "question": question_data['Question'],
                "correct_answer": question_data['Correct Answer'],
                "correct_explanation": question_data['Explanation'],
                "model_responses": {
                    "deepseek": {
                        "full_response": r1_response,
                        "answer": r1_answer,
                        "reasoning": reasoning,
                        "grade": None
                    },
                    "claude": {
                        "standalone": {
                            "full_response": claude_response,
                            "answer": claude_answer,
                            "grade": None
                        },
                        "with_reasoning": {
                            "full_response": claude_r1_response,
                            "answer": claude_r1_answer,
                            "grade": None
                        }
                    }
                },
                "metadata": {
                    "subdomain": question_data['High-level domain'],
                    "high_level_domain": question_data['High-level domain'],
                    "difficulty": question_data["Writer's Difficulty Estimate"]
                }
            }
            
            # Append single result to JSONL file
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
            
        except Exception as e:
            metrics["errors"] += 1
            tqdm.write(f"\nError processing example {record_id}: {str(e)}")
            success = False
            continue
    
    # After all processing, upload entire dataset to HuggingFace
    if success:
        try:
            print("\nUploading complete dataset to HuggingFace...")
            dataset = load_dataset('json', data_files=jsonl_path)
            dataset.push_to_hub(
                repo_id=Config.HF_DATASET_REPO,
                token=Config.HF_TOKEN,
                private=False,
                commit_message=f"Add batch of {len(data)} results"
            )
            print(f"✅ Successfully uploaded dataset to {Config.HF_DATASET_REPO}")
        except Exception as e:
            print(f"❌ Failed to upload to HuggingFace: {str(e)}")
            print(f"💾 Data saved locally to {jsonl_path}")
    
    print(f"\n✅ Benchmark completed!")
    print(f"💾 Results saved locally to: {jsonl_path}")
    print(f"\n📊 Processed {metrics['total_samples_processed']} samples")
    print(f"❌ Errors: {metrics['errors']}")
    print("\n🌍 Samples by domain:")
    for domain, count in metrics["domains"].items():
        print(f"  - {domain}: {count}")

if __name__ == "__main__":
    run_benchmark()