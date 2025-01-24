import random
from datetime import datetime
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

    def process_model_response(self, model_config: dict, prompt: str) -> tuple[dict, dict, float, dict]:
        """Process responses for a single model configuration and calculate costs"""
        responses = {}
        token_usage = {}
        total_cost = 0.0
        costs_breakdown = {}  # Track costs by group/model
        
        model_key = model_config["model_key"]
        if "deepseek" in model_key.lower():
            # DeepSeek specific handling
            reasoning, response, tokens = self.clients.get_deepseek_response(prompt)
            responses = {
                "answer": self.response_processor.extract_answer(response),
                "full_response": response,
                "reasoning": reasoning,
                "grade": None
            }
            token_usage = {model_key: tokens}
            cost = self.cost_manager.calculate_cost(tokens, model_config)
            total_cost = cost
            costs_breakdown[model_key] = cost
            
        else:
            # Claude-style handling with groups
            for group in model_config["token_groups"]:
                if group["type"] == "grouped":
                    key = group["label"].lower().replace(" ", "_")
                    group_key = f"{model_key}_{key}"  # Create consistent group key
                    
                    if key == "with_reasoning":
                        reasoning, _, deepseek_tokens = self.clients.get_deepseek_response(prompt)
                        modified_prompt = f"{prompt}\n\n<reasoning>{reasoning}</reasoning>"
                        response, tokens = self.clients.get_claude_response(modified_prompt)
                    else:
                        response, tokens = self.clients.get_claude_response(prompt)
                    
                    responses[key] = {
                        "answer": self.response_processor.extract_answer(response),
                        "full_response": response,
                        "grade": None
                    }
                    token_usage[group_key] = tokens  # Use consistent group key
                    group_cost = self.cost_manager.calculate_cost(tokens, model_config)
                    costs_breakdown[group_key] = group_cost  # Store cost with consistent key
                    total_cost += group_cost
        
        return responses, token_usage, total_cost, costs_breakdown

    def prepare_result(self, question_data: Dict[str, Any], responses: Dict[str, Any],
                      token_usage: Dict[str, Any], costs: Dict[str, float]) -> Dict[str, Any]:
        """Prepare the result record for storage"""
        return {
            "record_id": str(question_data['Record ID']),
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
                "total": sum(costs.values())  # Total cost for this record
            }
        }

    def process_batch(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Process a batch of questions"""
        batch_buffer = []
        
        for question_data in data:
            try:
                prompt = self.dataset.get_formatted_prompt(question_data)
                
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
                    
                    if "deepseek" in model_config["model_key"].lower():
                        self.cost_manager.track_usage(
                            model_key=model_config["model_key"],
                            group_key="base",  
                            tokens=model_tokens
                        )
                    else:
                        for group_key, tokens in model_tokens.items():
                            self.cost_manager.track_usage(model_config["model_key"], group_key, tokens)

                result = self.prepare_result(question_data, responses, token_usage, costs)
                
                # Update metrics and save result
                self.cost_manager.increment_processed()
                self.cost_manager.update_domain_stats(question_data['High-level domain'])
                self.io_manager.save_result(result, output_path)
                
                # Batch upload handling
                batch_buffer.append(result)
                if len(batch_buffer) >= Config.BATCH_SIZE:
                    self.io_manager.upload_to_huggingface(output_path, len(batch_buffer))
                    batch_buffer = []

            except Exception as e:
                self.cost_manager.increment_errors()
                print(f"Error processing {question_data['Record ID']}: {str(e)}")

        # Upload remaining records in buffer
        if batch_buffer:
            self.io_manager.upload_to_huggingface(output_path, len(batch_buffer))

    def run(self):
        """Main benchmark execution flow"""
        # Load and prepare dataset
        data = self.io_manager.load_dataset()
        
        if not data:
            print("All samples already processed! No new API calls needed.")
            
            # Optionally try to consolidate and upload existing results
            consolidated_path = self.io_manager.consolidate_jsonl_files()
            if consolidated_path:
                print("\nAttempting to upload consolidated results to HuggingFace...")
                self.io_manager.upload_to_huggingface(consolidated_path, len(self.io_manager.get_processed_record_ids()))
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

        # Consolidate and upload final results
        consolidated_path = self.io_manager.consolidate_jsonl_files()
        if consolidated_path:
            self.io_manager.upload_to_huggingface(consolidated_path, self.cost_manager.metrics.processed)

        # Print final report
        self.cost_manager.print_final_report()

def main():
    orchestrator = BenchmarkOrchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()
