import json
import os
from typing import Dict, Any
from dataclasses import dataclass, field

from .config import Config

@dataclass
class CostMetrics:
    processed: int = 0
    errors: int = 0
    domains: Dict[str, int] = field(default_factory=dict)
    costs: Dict[str, float] = field(default_factory=lambda: {"total": 0.0}) 

class CostManager:
    def __init__(self, config_path: str = None):
        self.metrics = CostMetrics()
        
        # Use provided config path or default to root directory
        if config_path is None:
            # Get the root directory by going up two levels from the current file
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(root_dir, "config.json")
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json file with model specifications at {config_path}")
            
        try:
            with open(config_path) as f:
                config_data = json.load(f)
                if "models" not in config_data:
                    raise ValueError("Invalid config.json: missing 'models' key")
                self.model_config = config_data["models"]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config.json: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading config.json: {str(e)}")

    def estimate_cost(self, num_samples: int) -> Dict[str, Any]:
        """Estimate API costs using config.json"""
        results = {
            "estimated_costs": {},
            "token_breakdown": {},
            "total_tokens": 0,
            "pricing_rates": {},
            "total_input_tokens": 0,
            "total_output_tokens": 0
        }

        total_cost = total_input = total_output = 0

        # Initialize estimated_costs and token_breakdown for all models and groups
        for model in self.model_config:
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

        for model in self.model_config:
            model_key = model["model_key"]
            token_groups = model["token_groups"]
            rates = model["cost_per_million"]
            
            for group in token_groups:
                if group["type"] == "grouped":
                    # Generate unique key for grouped entries
                    group_key = f"{model_key}_{group['label'].lower().replace(' ', '_')}"
                    input_key = group["input"]
                    output_key = group["output"]
                    
                    input_tokens = model["tokens"]["input"]
                    output_tokens = model["tokens"]["output"]
                    
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

    def calculate_cost(self, tokens: Dict[str, int], model_config: dict) -> float:
        """Calculate cost using token counts and model config rates."""
        # Always use standard input/output keys
        input_tokens = tokens.get("input", 0)
        output_tokens = tokens.get("output", 0)
        
        input_rate = model_config["cost_per_million"]["input"] / 1_000_000
        output_rate = model_config["cost_per_million"]["output"] / 1_000_000
        
        # Validation for unexpected zero counts
        try:
            cost = (input_tokens * input_rate) + (output_tokens * output_rate)
            return round(cost, 4)
        except TypeError as e:
            raise

    def track_usage(self, model_key: str, group_key: str, tokens: Dict[str, int]) -> None:
        """Track token usage and update costs for a model/group"""
        try:
            # Normalize token format
            actual_tokens = {
                "input": tokens.get("input", 0),
                "output": tokens.get("output", 0)
            }
                
            model_config = next(m for m in self.model_config if m["model_key"] == model_key)
            cost = self.calculate_cost(actual_tokens, model_config)
            
            # Handle grouped vs single models
            cost_key = f"{model_key}_{group_key}" if group_key else model_key
            
            self.metrics.costs.setdefault(cost_key, 0.0)
            self.metrics.costs[cost_key] += cost
            self.metrics.costs["total"] += cost

        except Exception as e:
            print(f"ERROR in track_usage: {str(e)}")

    def update_domain_stats(self, domain: str) -> None:
        """Update domain statistics"""
        self.metrics.domains[domain] = self.metrics.domains.get(domain, 0) + 1

    def increment_processed(self) -> None:
        """Increment processed count"""
        self.metrics.processed += 1

    def increment_errors(self) -> None:
        """Increment error count"""
        self.metrics.errors += 1

    def print_cost_estimate(self, estimate: Dict[str, Any]) -> None:
        """Print cost estimate in a structured format."""
        print("\n=== Cost Estimate ===")
        for model in self.model_config:
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

    def print_final_report(self) -> None:
        """Print final benchmark report with costs and statistics"""
        print(f"\n🎯 Benchmark Complete")
        print(f"📦 Processed: {self.metrics.processed}")
        print(f"❌ Errors: {self.metrics.errors}")
        print("\n💰 Actual Costs:")
        
        for model in self.model_config:
            model_key = model["model_key"]
            display_name = model["display_name"]
            
            # Handle ALL token groups (grouped and single)
            for group in model["token_groups"]:
                cost_key = None
                
                if group["type"] == "grouped":
                    group_label = group['label'].lower().replace(" ", "_")
                    cost_key = f"{model_key}_{group_label}"
                    print_label = f"{display_name} ({group['label']})"
                else:
                    # Single-entry model (e.g., DeepSeek)
                    cost_key = model_key
                    print_label = display_name
                    # Avoid duplicate prints for single-group models
                    if any(g["type"] == "grouped" for g in model["token_groups"]):
                        continue
                
                # Only print if cost exists and is > 0
                if cost_key in self.metrics.costs and self.metrics.costs[cost_key] > 0.0:
                    print(f"  {print_label}: ${self.metrics.costs[cost_key]:.4f}")  
        
        print(f"  Total Cost: ${self.metrics.costs['total']:.4f}")
        print("\n🌍 Domain Distribution:")
        for domain, count in self.metrics.domains.items():
            print(f"  - {domain}: {count}")