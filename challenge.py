# Standard library imports
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Tuple

# Third-party imports
import pandas as pd
from anthropic import Anthropic
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
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
    load_dotenv()
    
    MAX_SAMPLES: Optional[int] = 1  # Set to None to run on all samples
    DATASET_NAME: str = "iDavidRein/gpqa"
    DATASET_SPLIT: str = "train"  # Only split available in GPQA
    RESULTS_DIR: str = "benchmark_results"
    
    # API configuration
    DEEPSEEK_API_KEY: str = os.getenv('DEEPSEEK_API_KEY')
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY')
    HF_TOKEN: str = os.getenv('HF_TOKEN')  # Add Hugging Face token
    
    # Validate API keys
    if not all([DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN]):
        raise ValueError("Missing API keys. Please check your .env file.")
    
    # Authenticate with Hugging Face
    login(token=HF_TOKEN)
    
    # Model configuration
    DEEPSEEK_MODEL: str = "deepseek-reasoner"
    CLAUDE_MODEL: str = "claude-3-5-haiku-20241022"
    
    # Retry configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 2

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
        random.seed(42)  # For reproducibility
        data = random.sample(list(data), Config.MAX_SAMPLES)
        num_samples = Config.MAX_SAMPLES
    else:
        num_samples = total_samples
    
    # Show cost estimate
    cost_estimate = estimate_cost(num_samples)
    print("\nEstimated costs:")
    print(f"DeepSeek Reasoner: ${cost_estimate['estimated_costs']['deepseek_reasoner']:.2f}")
    print(f"Claude Sonnet: ${cost_estimate['estimated_costs']['claude_sonnet']:.2f}")
    print(f"Total: ${cost_estimate['estimated_costs']['total']:.2f}")
    print(f"\nEstimated total tokens: {cost_estimate['token_estimates']['total_tokens']:,}")
    print("\nPricing (per 1M tokens):")
    print("DeepSeek Reasoner:")
    print(f"  Input: ${cost_estimate['pricing_info']['deepseek_reasoner']['input']}")
    print(f"  Output: ${cost_estimate['pricing_info']['deepseek_reasoner']['output']}")
    print("Claude Sonnet:")
    print(f"  Input: ${cost_estimate['pricing_info']['claude_sonnet']['input']}")
    print(f"  Output: ${cost_estimate['pricing_info']['claude_sonnet']['output']}")
    
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
                "correct_answer": example['Correct Answer'],  # Fixed field name
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

def estimate_cost(num_samples: int) -> dict:
    """Estimate API costs for running the benchmark."""
    
    # Load pricing configuration
    pricing_config_path = "pricing_config.json"
    if not os.path.exists(pricing_config_path):
        raise FileNotFoundError(f"Missing {pricing_config_path} file")
    
    with open(pricing_config_path, 'r') as f:
        pricing_config = json.load(f)
    
    # Extract configuration values
    DEEPSEEK_TOKENS = pricing_config["deepseek_reasoner"]["tokens"]
    CLAUDE_TOKENS = pricing_config["claude_sonnet"]["tokens"]
    DEEPSEEK_REASONER_COST = pricing_config["deepseek_reasoner"]["cost_per_million"]
    CLAUDE_SONNET_COST = pricing_config["claude_sonnet"]["cost_per_million"]
    
    # Calculate costs per model
    deepseek_cost = num_samples * (
        (DEEPSEEK_TOKENS["input"] * DEEPSEEK_REASONER_COST["input"] / 1_000_000) +
        (DEEPSEEK_TOKENS["output"] * DEEPSEEK_REASONER_COST["output"] / 1_000_000)
    )
    
    claude_cost = num_samples * (
        # Standalone Claude costs
        (CLAUDE_TOKENS["standalone_input"] * CLAUDE_SONNET_COST["input"] / 1_000_000) +
        (CLAUDE_TOKENS["standalone_output"] * CLAUDE_SONNET_COST["output"] / 1_000_000) +
        # Claude with reasoning costs
        (CLAUDE_TOKENS["with_reasoning_input"] * CLAUDE_SONNET_COST["input"] / 1_000_000) +
        (CLAUDE_TOKENS["with_reasoning_output"] * CLAUDE_SONNET_COST["output"] / 1_000_000)
    )
    
    total_cost = deepseek_cost + claude_cost
    
    total_tokens = num_samples * (
        # DeepSeek tokens
        DEEPSEEK_TOKENS["input"] + DEEPSEEK_TOKENS["output"] +
        # Claude standalone tokens
        CLAUDE_TOKENS["standalone_input"] + CLAUDE_TOKENS["standalone_output"] +
        # Claude with reasoning tokens
        CLAUDE_TOKENS["with_reasoning_input"] + CLAUDE_TOKENS["with_reasoning_output"]
    )
    
    return {
        "num_samples": num_samples,
        "estimated_costs": {
            "deepseek_reasoner": round(deepseek_cost, 2),
            "claude_sonnet": round(claude_cost, 2),
            "total": round(total_cost, 2)
        },
        "token_estimates": {
            "deepseek": DEEPSEEK_TOKENS,
            "claude": CLAUDE_TOKENS,
            "total_tokens": int(total_tokens)
        },
        "pricing_info": {
            "deepseek_reasoner": DEEPSEEK_REASONER_COST,
            "claude_sonnet": CLAUDE_SONNET_COST
        }
    }

if __name__ == "__main__":
    run_benchmark()