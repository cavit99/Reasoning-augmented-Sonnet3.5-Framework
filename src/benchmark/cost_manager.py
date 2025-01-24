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
    costs: Dict[str, float] = field(default_factory=lambda: {
        "deepseek_reasoner": 0.0,
        "claude_sonnet_standalone": 0.0,
        "claude_sonnet_with_reasoning": 0.0,
        "total": 0.0
    })

class CostManager:
    def __init__(self):
        self.metrics = CostMetrics()
        
        # Validate config.json exists and load it
        if not os.path.exists("config.json"):
            raise FileNotFoundError("Missing config.json file with model specifications")
            
        try:
            with open("config.json") as f:
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

    def calculate_cost(self, tokens: Dict[str, int], model_config: dict) -> float:
        """Calculate cost using token counts and model config rates."""
        input_rate = model_config["cost_per_million"]["input"] / 1_000_000
        output_rate = model_config["cost_per_million"]["output"] / 1_000_000
        return (tokens["input"] * input_rate) + (tokens["output"] * output_rate)

    def track_usage(self, model_key: str, group_key: str, tokens: Dict[str, int]) -> None:
        """Track token usage and update costs for a model/group"""
        try:
            model_config = next(m for m in self.model_config if m["model_key"] == model_key)
            cost = self.calculate_cost(tokens, model_config)
            
            # Create consistent cost key
            cost_key = f"{model_key}_{group_key}" if group_key else model_key
            
            # Initialize if not exists
            if cost_key not in self.metrics.costs:
                self.metrics.costs[cost_key] = 0.0
                
            self.metrics.costs[cost_key] += cost
            self.metrics.costs["total"] += cost

        except KeyError as e:
            print(f"Warning: Missing token key {str(e)} in tracking for {model_key}")
        except StopIteration:
            print(f"Warning: Unknown model key {model_key} in tracking")

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
        print(f"  DeepSeek Total: ${self.metrics.costs['deepseek_reasoner']:.4f}")
        print(f"  Claude Standalone: ${self.metrics.costs['claude_sonnet_standalone']:.4f}")
        print(f"  Claude with Reasoning: ${self.metrics.costs['claude_sonnet_with_reasoning']:.4f}")
        print(f"  Total Cost: ${self.metrics.costs['total']:.4f}")
        print("\n🌍 Domain Distribution:")
        for domain, count in self.metrics.domains.items():
            print(f"  - {domain}: {count}")
