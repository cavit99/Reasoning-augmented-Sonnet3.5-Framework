import random
from tqdm import tqdm
from typing import Dict, Any, List

from .config import Config
from .data_models import GPQADataset
from .clients import LLMClients, ResponseProcessor
from .cost_manager import CostManager
from .io_manager import IOManager

class BenchmarkOrchestrator:
    def __init__(self):
        Config.validate()
        self.clients = LLMClients()
        self.cost_manager = CostManager()
        self.io_manager = IOManager()
        self.dataset = GPQADataset()
        self.response_processor = ResponseProcessor()

    def process_model_response(self, model_config: dict, prompt: str, 
                         use_cached_reasoning: bool = False,
                         cached_reasoning: str = None) -> tuple[dict, dict, float, dict]:
        """Now handles cached reasoning and partial processing"""
        responses = {}
        token_usage = {}
        total_cost = 0.0
        costs_breakdown = {}
        
        model_key = model_config["model_key"]
        if "deepseek" in model_key.lower():
            if use_cached_reasoning:
                # Return empty response if already processed
                return {}, {}, 0.0, {}
            else:
                # Original DeepSeek processing with model_name passed from config
                reasoning, response, tokens = self.clients.get_deepseek_response(
                    prompt, 
                    model_config["model_name"]  # Pass model_name from config
                )
                responses = {
                    "answer": self.response_processor.extract_answer(response),
                    "full_response": response,
                    "reasoning": reasoning,
                    "grade": None
                }
                formatted_tokens = {
                    "input": tokens["input"],
                    "output": tokens["output"]
                }
                token_usage = {model_key: formatted_tokens}
                cost = self.cost_manager.calculate_cost(tokens, model_config)
                total_cost = cost
                costs_breakdown[model_key] = cost
            
        else:
            # For Claude models using cached reasoning
            if use_cached_reasoning and cached_reasoning:
                # Skip DeepSeek call
                reasoning = cached_reasoning
                deepseek_tokens = {"input": 0, "output": 0}
            else:
                # Check if we already have DeepSeek reasoning
                if "deepseek_reasoner" in responses:  
                    reasoning = responses["deepseek_reasoner"]["reasoning"]
                    deepseek_tokens = {"input": 0, "output": 0}  # No new call
                else:
                    # Original DeepSeek call
                    reasoning, _, deepseek_tokens = self.clients.get_deepseek_response(prompt)

            # Claude-style handling with groups
            for group in model_config["token_groups"]:
                if group["type"] == "grouped":
                    key = group["label"].lower().replace(" ", "_")
                    
                    if key == "with_reasoning":
                        modified_prompt = f"{prompt}\n\n<reasoning>{reasoning}</reasoning>"
                        response, tokens = self.clients.get_claude_response(modified_prompt, model_config["model_name"])
                    else:
                        response, tokens = self.clients.get_claude_response(prompt, model_config["model_name"])
                    
                    responses[key] = {
                        "answer": self.response_processor.extract_answer(response),
                        "full_response": response,
                        "grade": None
                    }
                    token_usage[f"{model_key}_{key}"] = tokens  # Include model_key here for token_usage
                    group_cost = self.cost_manager.calculate_cost(tokens, model_config)
                    costs_breakdown[f"{model_key}_{key}"] = group_cost
                    total_cost += group_cost
                    
                    # Track Claude's usage with correct group_key
                    self.cost_manager.track_usage(
                        model_config["model_key"],
                        key,  # Pass raw group label (e.g., "with_reasoning")
                        tokens
                    )
        
        return responses, token_usage, total_cost, costs_breakdown
    
    def prepare_result(self, question_data: Dict[str, Any], responses: Dict[str, Any],
                    token_usage: Dict[str, Any], costs: Dict[str, float]) -> Dict[str, Any]:
        """Prepare the result record for storage"""
        result = {
            "record_id": str(question_data['Record ID']),
            "question": question_data['Question'],
            "correct_answer": question_data['Correct Answer'],
            "correct_explanation": question_data['Explanation'],
            "metadata": {
                "difficulty": question_data.get("Writer's Difficulty Estimate", "unknown"),
                "high_level_domain": question_data["High-level domain"],
                "subdomain": question_data.get("Subdomain", "unknown")
            },
            "token_usage": token_usage,
            "costs": {
                "total": round(sum(costs.values()), 4),
                **{k: round(v, 4) for k, v in costs.items()}
            }
        }

        # Extract answers and grades to top-level fields
        for model_key, model_data in responses.items():
            if isinstance(model_data, dict) and "answer" in model_data:
                # Handle non-grouped models (e.g., deepseek)
                result[f"{model_key}_answer"] = model_data.pop("answer")
                result[f"{model_key}_grade"] = model_data.pop("grade")
            else:
                # Handle grouped responses (e.g., Claude)
                for group_key, group_data in model_data.items():
                    result[f"{model_key}_{group_key}_answer"] = group_data.pop("answer")
                    result[f"{model_key}_{group_key}_grade"] = group_data.pop("grade")

        # Add cleaned model_responses (without answers/grades)
        result["model_responses"] = responses
        
        return result

    def process_batch(self, data: list, output_path: str) -> None:
        # Split into update vs new processing
        existing_to_update, new_to_process = data
        
        # Process new records (full)
        for record in tqdm(new_to_process, desc="New records"):
            self._process_full_record(record, output_path)
        
        # Update existing records (partial)
        for record_data in tqdm(existing_to_update, desc="Updating existing"):
            self._update_existing_record(
                record_data['base_data'],
                record_data['existing'],
                record_data['missing_models'],
                output_path
            )

    def _update_existing_record(self, base_data, existing_record, missing_models, output_path):
        """Add missing models to existing record"""
        try:
            # Reuse cached DeepSeek reasoning if available
            cached_reasoning = existing_record.get('deepseek_reasoner_reasoning')
            prompt = self.dataset.get_formatted_prompt(base_data)
            
            # Process only missing models
            for model_config in self.cost_manager.model_config:
                if model_config["model_key"] not in missing_models:
                    continue
                    
                # Modified process_model_response that skips existing work
                model_resp, model_tokens, cost, cost_breakdown = self.process_model_response(
                    model_config, 
                    prompt,
                    use_cached_reasoning=True,
                    cached_reasoning=cached_reasoning
                )
                
                # Merge new data into existing record
                updated_record = self._merge_responses(
                    existing_record,
                    model_config["model_key"],
                    model_resp,
                    model_tokens,
                    cost_breakdown
                )
                
                # Save updated record
                self.io_manager.save_updated_record(updated_record, output_path)

        except Exception as e:
            self.cost_manager.increment_errors()
            print(f"Error updating record {base_data['Record ID']}: {str(e)}")

    def _process_full_record(self, record: Dict[str, Any], output_path: str) -> None:
        """Process a new record with all models"""
        try:
            prompt = self.dataset.get_formatted_prompt(record)
            
            responses = {}
            token_usage = {}
            costs = {}
            
            for model_config in self.cost_manager.model_config:
                model_resp, model_tokens, cost, cost_breakdown = self.process_model_response(
                    model_config, prompt
                )
                responses[model_config["model_key"]] = model_resp
                token_usage.update(model_tokens)
                costs.update(cost_breakdown)
                
            result = self.prepare_result(record, responses, token_usage, costs)
            self.io_manager.save_result(result, output_path)
            
        except Exception as e:
            self.cost_manager.increment_errors()
            print(f"Error processing new record {record['Record ID']}: {str(e)}")

    def _merge_responses(self, existing: dict, model_key: str, 
                    new_responses: dict, tokens: dict, costs: dict) -> dict:
        """Merge new model responses into existing record"""
        merged = existing.copy()
        
        # Update answers
        for key, value in new_responses.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    merged[f"{model_key}_{key}_{subkey}"] = subvalue
            else:
                merged[f"{model_key}_{key}"] = value
                
        # Update tokens and costs
        merged["token_usage"].update(tokens)
        merged["costs"].update({k: round(v, 4) for k, v in costs.items()})
        merged["costs"]["total"] = round(sum(merged["costs"].values()), 4)
        
        return merged


    def run(self):
        """Main benchmark execution flow"""
        # Load and prepare dataset
        data = self.io_manager.load_dataset()
        
        if not data:
            print("All samples already processed! No new API calls needed.")
            return

        # Sample data if MAX_SAMPLES is set
        if Config.MAX_SAMPLES:
            random.seed(99)
            data = random.sample(data, min(Config.MAX_SAMPLES, len(data)))
            print(f"\nSelected {len(data)} samples for processing")
        
        # Show cost estimate and get confirmation
        cost_estimate = self.cost_manager.estimate_cost(len(data))
        self.cost_manager.print_cost_estimate(cost_estimate)
        if input("\nProceed? (Y/n): ").lower() not in ('', 'y'):
            print("Aborted.")
            return

        # Process data
        output_path = Config.get_output_path()
        self.process_batch(data, output_path)

        # Consolidate results after processing
        consolidated_path = self.io_manager.consolidate_jsonl_files()
        if consolidated_path:
            print(f"\nConsolidated results saved to: {consolidated_path}")

        # Print final report
        self.cost_manager.print_final_report()

def main():
    orchestrator = BenchmarkOrchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()
