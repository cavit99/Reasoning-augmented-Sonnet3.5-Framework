import os
from typing import Tuple, List, Dict
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from datasets import load_dataset
import random

class TokenCounterConfig:
    """Configuration settings aligned with challenge.py"""
    MAX_SAMPLES = 1
    DATASET_NAME = "iDavidRein/gpqa"
    DATASET_CONFIG = "gpqa_diamond"
    DATASET_SPLIT = "train"
    
    # Models match challenge.py except keeping Haiku
    DEEPSEEK_MODEL = "deepseek-reasoner"
    CLAUDE_MODEL = "claude-3-5-haiku-20241022"

    # Debug flag to control API call logging
    DEBUG_MODE = True

class TokenCounter:
    """Token counter for GPQA dataset with non-streaming approach"""
    
    def __init__(self):
        self.deepseek_client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.results: List[Dict[str, Dict[str, int]]] = []
        
        # Verify API connections
        if TokenCounterConfig.DEBUG_MODE:
            self._verify_api_connections()
            
        self.model_configs = {
            "deepseek": {
                "client": self.deepseek_client,
                "model": TokenCounterConfig.DEEPSEEK_MODEL,
                "create_completion": self._create_deepseek_completion
            },
            "claude_standalone": {
                "client": self.anthropic_client,
                "model": TokenCounterConfig.CLAUDE_MODEL,
                "create_completion": self._create_claude_completion
            },
            "claude_with_reasoning": {
                "client": self.anthropic_client,
                "model": TokenCounterConfig.CLAUDE_MODEL,
                "create_completion": self._create_claude_with_reasoning_completion
            }
        }

    def _verify_api_connections(self):
        """Verify API connections with a simple test"""
        print("\n=== Verifying API Connections ===")
        try:
            # Test DeepSeek
            print("Testing DeepSeek API connection...")
            self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Say 'test' in one word."}],
                timeout=30
            )
            print("✓ DeepSeek API connection successful")
            
            # Test Claude
            print("Testing Claude API connection...")
            self.anthropic_client.messages.create(
                model=TokenCounterConfig.CLAUDE_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'test' in one word."}]
            )
            print("✓ Claude API connection successful")
            
        except Exception as e:
            print(f"❌ API connection verification failed: {str(e)}")
            raise

    def _create_deepseek_completion(self, prompt: str, reasoning_content: str = None) -> dict:
        if TokenCounterConfig.DEBUG_MODE:
            print("\n=== DeepSeek API Call ===")
            print("Input prompt:", prompt)
            print("Attempting API call...")
            
        try:
            response = self.deepseek_client.chat.completions.create(
                model=TokenCounterConfig.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=180  # Add timeout to prevent indefinite hanging
            )
            
            if TokenCounterConfig.DEBUG_MODE:
                print("API call successful")
                print("\nResponse:", response.choices[0].message.content)
                print(f"Input tokens: {response.usage.prompt_tokens}")
                print(f"Output tokens: {response.usage.completion_tokens}")
                
            return {
                "response": response,
                "usage": {"input": response.usage.prompt_tokens, 
                         "output": response.usage.completion_tokens}
            }
        except Exception as e:
            print(f"DeepSeek API call failed: {str(e)}")
            raise  # Re-raise the exception to be caught by the main error handler

    def _create_claude_completion(self, prompt: str, reasoning_content: str = None) -> dict:
        if TokenCounterConfig.DEBUG_MODE:
            print("\n=== Claude API Call ===")
            print("Input prompt:", prompt)
            
        response = self.anthropic_client.messages.create(
            model=TokenCounterConfig.CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        if TokenCounterConfig.DEBUG_MODE:
            print("\nResponse:", response.content[0].text)
            print(f"Input tokens: {response.usage.input_tokens}")
            print(f"Output tokens: {response.usage.output_tokens}")
            
        return {
            "response": response,
            "usage": {"input": response.usage.input_tokens, 
                     "output": response.usage.output_tokens}
        }

    def _create_claude_with_reasoning_completion(self, prompt: str, reasoning_content: str) -> dict:
        combined_prompt = f"{prompt}\n\n<reasoning>{reasoning_content}</reasoning>"
        return self._create_claude_completion(combined_prompt)

    def run_analysis(self) -> None:
        """Main analysis workflow"""
        if TokenCounterConfig.DEBUG_MODE:
            print("\n=== Starting Analysis ===")
            print(f"Dataset: {TokenCounterConfig.DATASET_NAME}/{TokenCounterConfig.DATASET_CONFIG}")
            print("Initializing API clients...")
            
        try:
            dataset = load_dataset(
                TokenCounterConfig.DATASET_NAME,
                TokenCounterConfig.DATASET_CONFIG,
                token=os.getenv('HF_TOKEN')
            )
            if TokenCounterConfig.DEBUG_MODE:
                print("Dataset loaded successfully")
                
            data = dataset[TokenCounterConfig.DATASET_SPLIT]
            
            if TokenCounterConfig.MAX_SAMPLES:
                random.seed(999)
                data = random.sample(list(data), TokenCounterConfig.MAX_SAMPLES)
                if TokenCounterConfig.DEBUG_MODE:
                    print(f"\nProcessing {len(data)} samples")

            for i, example in enumerate(data, 1):
                try:
                    if TokenCounterConfig.DEBUG_MODE:
                        print(f"\n--- Processing Sample {i}/{len(data)} ---")
                        print(f"Record ID: {example.get('Record ID', 'N/A')}")
                
                    prompt = self._format_prompt(example)
                    result = {}
                    
                    # Get DeepSeek response first for reasoning content
                    deepseek_result = self._create_deepseek_completion(prompt)
                    reasoning_content = deepseek_result["response"].choices[0].message.reasoning_content
                    
                    # Process all models
                    for model_name, config in self.model_configs.items():
                        completion = config["create_completion"](prompt, reasoning_content)
                        result[model_name] = completion["usage"]
                
                    self.results.append(result)
                    
                    if TokenCounterConfig.DEBUG_MODE:
                        print(f"\nSuccessfully processed sample {i}")
                
                except Exception as e:
                    print(f"Error processing record {example['Record ID']}: {str(e)}")
                    continue

            if TokenCounterConfig.DEBUG_MODE:
                print("\n=== Analysis Complete ===")
                
            self._print_statistics()
        except Exception as e:
            print(f"Analysis failed: {str(e)}")

    def _format_prompt(self, example: dict) -> str:
        """Match challenge.py's prompt format exactly"""
        return (
            f"{example['Question']}\n\n"
            "Please provide a detailed explanation of your reasoning, "
            "then state your final answer clearly enclosed in <answer>...</answer> XML tags."
        )

    def _print_statistics(self) -> None:
        """Print formatted token usage results"""
        totals = {
            "deepseek_input": sum(r["deepseek"]["input"] for r in self.results),
            "deepseek_output": sum(r["deepseek"]["output"] for r in self.results),
            "claude_standalone_input": sum(r["claude_standalone"]["input"] for r in self.results),
            "claude_standalone_output": sum(r["claude_standalone"]["output"] for r in self.results),
            "claude_reasoning_input": sum(r["claude_with_reasoning"]["input"] for r in self.results),
            "claude_reasoning_output": sum(r["claude_with_reasoning"]["output"] for r in self.results),
        }

        print("\n=== Token Usage Summary ===")
        print(f"Samples analyzed: {len(self.results)}")
        
        print("\nAverage per sample:")
        for k, v in totals.items():
            print(f"{k.replace('_', ' ').title()}: {v/len(self.results):.1f} tokens")
            
        print("\nTotal for batch:")
        for k, v in totals.items():
            print(f"{k.replace('_', ' ').title()}: {v} tokens")

if __name__ == "__main__":
    load_dotenv()
    TokenCounter().run_analysis()