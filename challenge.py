# Standard library imports
import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List

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
    BATCH_SIZE: int = 100  # Number of samples per HuggingFace upload
    
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
    
    # Dataset configuration
    HF_DATASET_REPO: str = "spawn99/GPQA-diamond-ClaudeR1"

class SchemaField:
    """Combines type validation and serialization format"""
    def __init__(self, py_type, hf_type=None):
        self.py_type = py_type
        self.hf_type = hf_type or self._default_hf_type(py_type)
        
    def _default_hf_type(self, t):
        type_map = {
            str: "string",
            int: "int64",
            float: "float64",
            bool: "bool",
            type(None): "null"
        }
        return type_map.get(t, "string")

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
    return os.path.join(Config.RESULTS_DIR, f"{base_name}.jsonl")

def extract_answer(text: str) -> Optional[str]:
    """Extract answer from between XML tags."""
    if not text:
        return None
    
    text = text.strip()
    if not text:
        return None
    
    if text in ('</answer>', '<answer>'):
        return None
    
    pattern = r'<answer[^>]*>(.*?)</answer>'
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

    def get_deepseek_response(self, prompt: str) -> Tuple[str, str, Dict[str, int]]:
        """Get response from DeepSeek with retry logic and token tracking."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.deepseek_client.chat.completions.create(
                    model=Config.DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return (
                    response.choices[0].message.reasoning_content or "",
                    response.choices[0].message.content or "",
                    {
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens
                    }
                )
                
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise
                time.sleep(Config.RETRY_DELAY)

    def get_claude_response(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        """Get response from Claude with retry logic and token tracking."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.anthropic_client.messages.create(
                    model=Config.CLAUDE_MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return (
                    response.content[0].text,
                    {
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens
                    }
                )
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise
                time.sleep(Config.RETRY_DELAY)

def estimate_cost(num_samples: int) -> Dict[str, Any]:
    """Estimate API costs using config.json"""
    with open("config.json") as f:
        config = json.load(f)
    
    results = {
        "estimated_costs": {},
        "token_breakdown": {},
        "total_tokens": 0,
        "pricing_rates": {}
    }

    total_cost = total_input = total_output = 0

    # Initialize estimated_costs and token_breakdown for all models and groups
    for model in config["models"]:
        model_key = model["model_key"]
        results["pricing_rates"][model_key] = model["cost_per_million"]
        
        # Handle grouped models
        if any(g["type"] == "grouped" for g in model["token_groups"]):
            for group in model["token_groups"]:
                group_key = f"{model_key}_{group['label'].lower().replace(' ', '_')}"
                results["estimated_costs"][group_key] = 0.0
                results["token_breakdown"][group_key] = {"input": 0, "output": 0}
        else:
            results["estimated_costs"][model_key] = 0.0
            results["token_breakdown"][model_key] = {"input": 0, "output": 0}

    for model in config["models"]:
        model_key = model["model_key"]
        token_groups = model["token_groups"]
        rates = model["cost_per_million"]
        
        for group in token_groups:
            if group["type"] == "grouped":
                # Generate unique key for grouped entries
                group_key = f"{model_key}_{group['label'].lower().replace(' ', '_')}"
                input_key = group["input"]
                output_key = group["output"]
                
                input_tokens = model["tokens"][input_key]
                output_tokens = model["tokens"][output_key]
                
                # Calculate costs for this group
                input_cost = (input_tokens * rates["input"] / 1_000_000) * num_samples
                output_cost = (output_tokens * rates["output"] / 1_000_000) * num_samples
                group_cost = input_cost + output_cost
                
                # Update group-specific entries
                results["estimated_costs"][group_key] += round(group_cost, 2)
                results["token_breakdown"][group_key]["input"] += input_tokens * num_samples
                results["token_breakdown"][group_key]["output"] += output_tokens * num_samples
                
                total_cost += group_cost
                total_input += input_tokens * num_samples
                total_output += output_tokens * num_samples
                
            else:  # Handle single-entry models
                input_key = group["input"]
                output_key = group["output"]
                
                input_tokens = model["tokens"][input_key]
                output_tokens = model["tokens"][output_key]
                
                input_cost = (input_tokens * rates["input"] / 1_000_000) * num_samples
                output_cost = (output_tokens * rates["output"] / 1_000_000) * num_samples
                model_cost = input_cost + output_cost

                results["estimated_costs"][model_key] += round(model_cost, 2)
                results["token_breakdown"][model_key]["input"] += input_tokens * num_samples
                results["token_breakdown"][model_key]["output"] += output_tokens * num_samples
                
                total_cost += model_cost
                total_input += input_tokens * num_samples
                total_output += output_tokens * num_samples

    # Update totals
    results["estimated_costs"]["total"] = round(total_cost, 2)
    results["total_tokens"] = total_input + total_output
    results["total_input_tokens"] = total_input
    results["total_output_tokens"] = total_output

    return results

def print_cost_estimate(estimate: Dict[str, Any]) -> None:
    """Print cost estimate in a structured format."""
    with open("config.json") as f:
        config = json.load(f)

    print("\n=== Cost Estimate ===")
    for model in config["models"]:
        if "claude" in model["model_key"].lower():
            # Split Claude costs by group
            for group in model["token_groups"]:
                group_name = f"{model['display_name']} ({group['label']})"
                cost = estimate["estimated_costs"][f"{model['model_key']}_{group['label'].lower().replace(' ', '_')}"]
                print(f"{group_name}: ${cost:.2f}")
        else:
            # Regular display for non-Claude models
            cost = estimate["estimated_costs"][model["model_key"]]
            print(f"{model['display_name']}: ${cost:.2f}")
    
    print(f"\nTotal Cost: ${estimate['estimated_costs']['total']:.2f}")

def get_processed_record_ids() -> set[str]:
    """Get set of already processed record IDs from local JSONL files."""
    processed_ids = set()
    
    try:
        for fname in os.listdir(Config.RESULTS_DIR):
            if fname.endswith('.jsonl'):
                with open(os.path.join(Config.RESULTS_DIR, fname)) as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            if "record_id" in record:
                                # Ensure record_id is stored as string
                                processed_ids.add(str(record["record_id"]))
                        except json.JSONDecodeError:
                            continue
    except FileNotFoundError:
        # Results directory doesn't exist yet
        pass
    except Exception as e:
        print(f"Warning: Error reading local files: {e}")
    
    return processed_ids

# Add this class definition before the validate_record function
class SchemaManager:
    @staticmethod
    def _get_config():
        """Load model config from JSON file"""
        with open("config.json") as f:
            return json.load(f)["models"]

    @staticmethod
    def get_record_schema():
        """Central definition of the record schema"""
        models = SchemaManager._get_config()
        return {
            "record_id": SchemaField(str),
            "question": SchemaField(str),
            "correct_answer": SchemaField(str),
            "correct_explanation": SchemaField(str),
            "model_responses": {
                "type": "nested",
                "structure": {
                    model["model_key"]: SchemaManager.get_model_response_structure(model)
                    for model in models
                }
            },
            "metadata": {
                "type": "nested", 
                "structure": {
                    "difficulty": SchemaField(str),
                    "high_level_domain": SchemaField(str),
                    "subdomain": SchemaField(str)
                }
            },
            "token_usage": {
                "type": "nested",
                "structure": SchemaManager.get_token_usage_structure()
            }
        }

    @staticmethod
    def get_model_response_structure(model_config):
        structure = {}
        for group in model_config["token_groups"]:
            if group["type"] == "grouped":
                key = group["label"].lower().replace(" ", "_")
                structure[key] = {
                    "answer": SchemaField(str),
                    "full_response": SchemaField(str),
                    "grade": SchemaField(type(None))
                }
            else:
                structure.update({
                    "answer": SchemaField(str),
                    "full_response": SchemaField(str),
                    "reasoning": SchemaField(str),
                    "grade": SchemaField(type(None))
                })
        return structure

    @staticmethod
    def get_token_usage_structure():
        structure = {}
        for model in SchemaManager._get_config():
            for group in model["token_groups"]:
                key = f"{model['model_key']}"
                if group["type"] == "grouped":
                    key += f"_{group['label'].lower().replace(' ', '_')}"
                structure[key] = {
                    "input": SchemaField(int),
                    "output": SchemaField(int)
                }
        return structure

def validate_record(record: dict) -> bool:
    """Validate record structure using central schema."""
    schema = SchemaManager.get_record_schema()
    
    def check_structure(data, schema_node):
        if isinstance(schema_node, SchemaField):
            return isinstance(data, schema_node.py_type)
            
        if isinstance(schema_node, dict):
            if "type" in schema_node and schema_node["type"] == "nested":
                if not isinstance(data, dict):
                    return False
                return all(
                    check_structure(data.get(k), v)
                    for k, v in schema_node["structure"].items()
                )
            # Handle regular dict structure
            if not isinstance(data, dict):
                return False
            return all(
                check_structure(data.get(k), v)
                for k, v in schema_node.items()
            )
            
        return False
    
    return check_structure(record, schema)

def calculate_cost_from_tokens(tokens: Dict[str, int], model_config: dict) -> float:
    """Calculate cost using token counts and model config rates."""
    input_rate = model_config["cost_per_million"]["input"] / 1_000_000
    output_rate = model_config["cost_per_million"]["output"] / 1_000_000
    return (tokens["input"] * input_rate) + (tokens["output"] * output_rate)

def process_model_response(runner: ModelRunner, model_config: dict, prompt: str) -> Tuple[dict, dict, float, dict]:
    """Process responses for a single model configuration and calculate costs"""
    responses = {}
    token_usage = {}
    total_cost = 0.0
    costs_breakdown = {}  # Track costs by group/model
    
    model_key = model_config["model_key"]
    if "deepseek" in model_key.lower():
        # DeepSeek specific handling
        reasoning, response, tokens = runner.get_deepseek_response(prompt)
        responses = {
            "answer": extract_answer(response),
            "full_response": response,
            "reasoning": reasoning,
            "grade": None
        }
        token_usage = {model_key: tokens}
        cost = calculate_cost_from_tokens(tokens, model_config)
        total_cost = cost
        costs_breakdown[model_key] = cost
        
    else:
        # Claude-style handling with groups
        for group in model_config["token_groups"]:
            if group["type"] == "grouped":
                key = group["label"].lower().replace(" ", "_")
                group_key = f"{model_key}_{key}"  # Create consistent group key
                
                if key == "with_reasoning":
                    reasoning, _, deepseek_tokens = runner.get_deepseek_response(prompt)
                    modified_prompt = f"{prompt}\n\n<reasoning>{reasoning}</reasoning>"
                    response, tokens = runner.get_claude_response(modified_prompt)
                else:
                    response, tokens = runner.get_claude_response(prompt)
                
                responses[key] = {
                    "answer": extract_answer(response),
                    "full_response": response,
                    "grade": None
                }
                token_usage[group_key] = tokens  # Use consistent group key
                group_cost = calculate_cost_from_tokens(tokens, model_config)
                costs_breakdown[group_key] = group_cost  # Store cost with consistent key
                total_cost += group_cost
    
    return responses, token_usage, total_cost, costs_breakdown

def consolidate_jsonl_files() -> Optional[str]:
    """Merge all JSONL files in RESULTS_DIR with deduplication and validation."""
    seen_ids = set()
    all_results = []
    error_count = 0
    
    for fname in os.listdir(Config.RESULTS_DIR):
        if fname.endswith('.jsonl'):
            with open(os.path.join(Config.RESULTS_DIR, fname)) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line)
                        if not validate_record(record):
                            raise ValueError("Invalid record structure")
                            
                        record_id = record['record_id']
                        
                        if record_id in seen_ids:
                            continue
                            
                        seen_ids.add(record_id)
                        all_results.append(record)
                        
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        error_count += 1
                        print(f"Invalid record in {fname} line {line_num}: {str(e)}")
                        continue
    
    if error_count > 0:
        print(f"Skipped {error_count} invalid/malformed records during consolidation")
    
    if not all_results:
        return None
    
    # Sort records by record_id for consistency
    all_results.sort(key=lambda x: x['record_id'])
    
    consolidated_path = os.path.join(Config.RESULTS_DIR, "consolidated.jsonl")
    with open(consolidated_path, 'w') as f:
        for res in all_results:
            f.write(json.dumps(res) + '\n')
            
    return consolidated_path

def upload_to_huggingface(jsonl_path: str, num_records: int) -> None:
    """Upload results with dynamic schema generation."""
    from datasets import Features, Value
    
    def build_features(schema_node):
        if isinstance(schema_node, SchemaField):
            return Value(schema_node.hf_type)
        if isinstance(schema_node, dict):
            if "type" in schema_node and schema_node["type"] == "nested":
                return {k: build_features(v) for k, v in schema_node["structure"].items()}
            return {k: build_features(v) for k, v in schema_node.items()}
        return schema_node
    
    schema = SchemaManager.get_record_schema()
    features = Features(build_features(schema))
    
    try:
        existing_ds = load_dataset(Config.HF_DATASET_REPO, split="train")
    except Exception:
        existing_ds = None

    try:
        new_ds = load_dataset('json', data_files=jsonl_path, features=features, split='train')
        combined_ds = concatenate_datasets([existing_ds, new_ds]) if existing_ds else new_ds
        
        combined_ds.push_to_hub(
            repo_id=Config.HF_DATASET_REPO,
            token=Config.HF_TOKEN,
            private=False,
            commit_message=f"Add batch of {num_records} results"
        )
        print(f"✅ Uploaded {num_records} new records to {Config.HF_DATASET_REPO}")
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")

def run_benchmark():
    """Main benchmark execution flow."""
    setup_directories()
    processed_ids = get_processed_record_ids()
    print(f"Found {len(processed_ids)} processed records")

    # Load and filter dataset
    dataset = load_dataset(Config.DATASET_NAME, "gpqa_diamond", token=Config.HF_TOKEN)
    data = [d for d in dataset[Config.DATASET_SPLIT] if d['Record ID'] not in processed_ids]
    print(f"Found {len(data)} unprocessed records")
    
    if not data:
        print("All samples already processed! No new API calls needed.")
        
        # Optionally try to consolidate and upload existing results
        consolidated_path = consolidate_jsonl_files()
        if consolidated_path:
            print("\nAttempting to upload consolidated results to HuggingFace...")
            upload_to_huggingface(consolidated_path, len(processed_ids))
        return

    # Only show cost estimate if there are new samples to process
    if Config.MAX_SAMPLES:
        random.seed(99)
        data = random.sample(data, min(Config.MAX_SAMPLES, len(data)))
        print(f"\nSelected {len(data)} samples for processing")
    
    cost_estimate = estimate_cost(len(data))
    print_cost_estimate(cost_estimate)
    if input("\nProceed? (Y/n): ").lower() not in ('', 'y'):
        print("Aborted.")
        return

    # Initialize components
    gpqa_dataset = GPQADataset()
    runner = ModelRunner()
    batch_buffer = []
    jsonl_path = get_output_paths()
    metrics = {
        "processed": 0,
        "errors": 0,
        "domains": {},
        "start_time": datetime.now().isoformat(),
        "costs": {
            "deepseek_reasoner": 0.0,
            "claude_sonnet_standalone": 0.0,  # Use consistent keys
            "claude_sonnet_with_reasoning": 0.0,
            "total": 0.0
        }
    }

    # Processing loop
    for question_data in tqdm(data, total=len(data)):
        record_id = question_data['Record ID']
        try:
            prompt = gpqa_dataset.get_formatted_prompt(question_data)
            
            responses = {}
            token_usage = {}
            costs = {}
            total_cost = 0.0
            
            for model_config in SchemaManager._get_config():
                model_resp, model_tokens, cost, cost_breakdown = process_model_response(
                    runner, model_config, prompt
                )
                responses[model_config["model_key"]] = model_resp
                token_usage.update(model_tokens)
                costs.update(cost_breakdown)
                total_cost += cost
                
                # Update metrics using consistent keys
                if "deepseek" in model_config["model_key"].lower():
                    metrics["costs"]["deepseek_reasoner"] += cost
                else:
                    for group_key, group_cost in cost_breakdown.items():
                        metrics["costs"][group_key] += group_cost
                metrics["costs"]["total"] += cost

            result = {
                "record_id": str(record_id),
                "question": question_data['Question'],
                "correct_answer": question_data['Correct Answer'],
                "correct_explanation": question_data['Explanation'],
                "model_responses": responses,
                "metadata": {
                    "difficulty": question_data.get("Writer's Difficulty Estimate", "unknown"),
                    "high_level_domain": question_data["High-level domain"],
                    "subdomain": question_data.get("Subdomain", "unknown")
                },
                "token_usage": token_usage,
                "costs": {
                    **costs,  # Individual model/group costs
                    "total": total_cost  # Total cost for this record
                }
            }

            # Update local storage and metrics
            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
            
            metrics["processed"] += 1
            domain = question_data['High-level domain']
            metrics["domains"][domain] = metrics["domains"].get(domain, 0) + 1
            
            # Batch upload handling
            batch_buffer.append(result)
            if len(batch_buffer) >= Config.BATCH_SIZE:
                upload_to_huggingface(jsonl_path, len(batch_buffer))
                batch_buffer = []

        except Exception as e:
            metrics["errors"] += 1
            tqdm.write(f"Error processing {record_id}: {str(e)}")

    # Final upload of remaining records
    if batch_buffer:
        upload_to_huggingface(jsonl_path, len(batch_buffer))

    # Consolidate results and final upload
    consolidated_path = consolidate_jsonl_files()
    if consolidated_path:
        upload_to_huggingface(consolidated_path, metrics["processed"])

    # Print final report with costs
    print(f"\n🎯 Benchmark Complete")
    print(f"📦 Processed: {metrics['processed']}")
    print(f"❌ Errors: {metrics['errors']}")
    print(f"⏱️ Duration: {datetime.now() - datetime.fromisoformat(metrics['start_time'])}")
    print("\n💰 Actual Costs:")
    print(f"  DeepSeek Total: ${metrics['costs']['deepseek_reasoner']:.4f}")
    print(f"  Claude Standalone: ${metrics['costs']['claude_sonnet_standalone']:.4f}")
    print(f"  Claude with Reasoning: ${metrics['costs']['claude_sonnet_with_reasoning']:.4f}")
    print(f"  Total Cost: ${metrics['costs']['total']:.4f}")
    print("\n🌍 Domain Distribution:")
    for domain, count in metrics["domains"].items():
        print(f"  - {domain}: {count}")

if __name__ == "__main__":
    run_benchmark()