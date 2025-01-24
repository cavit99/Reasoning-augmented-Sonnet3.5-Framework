import os
from typing import Tuple, List, Dict, Any
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from datasets import load_dataset
import random

class TokenCounterConfig:
    """Configuration settings for token counting analysis."""
    MAX_SAMPLES = 5  # Small test batch for token counting
    DATASET_NAME = "ibragim-bad/arc_challenge"
    DATASET_SPLIT = "test"
    
    # Model names
    DEEPSEEK_MODEL = "deepseek-reasoner"
    CLAUDE_MODEL = "claude-3-5-haiku-20241022"

class TokenCounter:
    """
    Analyzes token usage for DeepSeek and Claude models on ARC Challenge dataset.
    Provides accurate token counts using direct API usage statistics.
    """
    
    def __init__(self):
        self.deepseek_client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.results: List[Dict[str, Dict[str, int]]] = []

    def run_analysis(self) -> None:
        """Runs token analysis on sample dataset and prints statistics."""
        dataset = load_dataset(TokenCounterConfig.DATASET_NAME, token=os.getenv('HF_TOKEN'))
        data = dataset[TokenCounterConfig.DATASET_SPLIT]
        
        if TokenCounterConfig.MAX_SAMPLES:
            random.seed(999)
            data = random.sample(list(data), TokenCounterConfig.MAX_SAMPLES)

        for i, example in enumerate(data, 1):
            print(f"\nProcessing Example {i}/{TokenCounterConfig.MAX_SAMPLES}")
            
            try:
                prompt = self.format_prompt(example['question'], example['choices'])
                
                # Process DeepSeek
                ds_reasoning, ds_answer, ds_input, ds_output, ds_token_details = self.get_deepseek_response(prompt)
                
                # Process Claude standalone
                claude1_response = self.get_claude_response(prompt)
                
                # Process Claude with reasoning
                combined_prompt = f"{prompt}\n<reasoning>{ds_reasoning}</reasoning>"
                claude2_response = self.get_claude_response(combined_prompt)
                
                # Store results
                self.results.append({
                    "deepseek": {
                        "input": ds_input,
                        "output": ds_output
                    },
                    "claude_standalone": {
                        "input": claude1_response.usage.input_tokens,
                        "output": claude1_response.usage.output_tokens
                    },
                    "claude_with_reasoning": {
                        "input": claude2_response.usage.input_tokens,
                        "output": claude2_response.usage.output_tokens
                    }
                })
                
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

        print("\n=== Final Results ===")
        self.print_statistics()

    def format_prompt(self, question: str, choices: Dict[str, Any]) -> str:
        """
        Format prompt to match the benchmark format.
        """
        choices_formatted = "\n".join([
            f"{label}. {text}" 
            for label, text in zip(choices['label'], choices['text'])
        ])
        return (
            f"Question: {question}\n\n"
            f"Choices:\n{choices_formatted}\n\n"
            "Please provide your reasoning step by step, then clearly state your final answer "
            "as a single letter (A, B, C, or D)."
        )

    def get_deepseek_response(self, prompt: str) -> Tuple[str, str, int, int, dict]:
        """
        Gets response and token counts from DeepSeek API.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Tuple of (reasoning_content, final_answer, input_tokens, output_tokens, token_details)
        """
        response = self.deepseek_client.chat.completions.create(
            model=TokenCounterConfig.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        # Get both parts of the response
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        
        # Get detailed token counts
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        token_details = response.usage.completion_tokens_details  # This shows the breakdown
        
        return reasoning_content, content, input_tokens, output_tokens, token_details

    def get_claude_response(self, prompt: str) -> Any:
        """
        Gets response from Claude API.
        
        Args:
            prompt: Input prompt text
            
        Returns:
            Claude API response object containing usage statistics
        """
        return self.anthropic_client.messages.create(
            model=TokenCounterConfig.CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

    def print_statistics(self) -> None:
        """Prints detailed token usage statistics from collected results."""
        total = len(self.results)
        if total == 0:
            print("No results collected")
            return

        # Initialize sums
        sums = {
            "deepseek_input": 0,
            "deepseek_output": 0,
            "claude_standalone_input": 0,
            "claude_standalone_output": 0,
            "claude_reasoning_input": 0,
            "claude_reasoning_output": 0
        }

        for result in self.results:
            sums["deepseek_input"] += result["deepseek"]["input"]
            sums["deepseek_output"] += result["deepseek"]["output"]
            sums["claude_standalone_input"] += result["claude_standalone"]["input"]
            sums["claude_standalone_output"] += result["claude_standalone"]["output"]
            sums["claude_reasoning_input"] += result["claude_with_reasoning"]["input"]
            sums["claude_reasoning_output"] += result["claude_with_reasoning"]["output"]

        print("\nToken Usage Statistics:")
        print(f"Samples analyzed: {total}")
        print("\nAverage per sample:")
        print(f"DeepSeek Input: {sums['deepseek_input']/total:.1f} tokens")
        print(f"DeepSeek Output: {sums['deepseek_output']/total:.1f} tokens")
        print(f"Claude Standalone Input: {sums['claude_standalone_input']/total:.1f} tokens")
        print(f"Claude Standalone Output: {sums['claude_standalone_output']/total:.1f} tokens")
        print(f"Claude with Reasoning Input: {sums['claude_reasoning_input']/total:.1f} tokens")
        print(f"Claude with Reasoning Output: {sums['claude_reasoning_output']/total:.1f} tokens")

        print("\nTotal for test batch:")
        print(f"DeepSeek Input: {sums['deepseek_input']} tokens")
        print(f"DeepSeek Output: {sums['deepseek_output']} tokens")
        print(f"Claude Standalone Input: {sums['claude_standalone_input']} tokens")
        print(f"Claude Standalone Output: {sums['claude_standalone_output']} tokens")
        print(f"Claude with Reasoning Input: {sums['claude_reasoning_input']} tokens")
        print(f"Claude with Reasoning Output: {sums['claude_reasoning_output']} tokens")

if __name__ == "__main__":
    load_dotenv()
    counter = TokenCounter()
    counter.run_analysis()
