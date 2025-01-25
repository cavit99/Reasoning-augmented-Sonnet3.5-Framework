import os
import re
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

from anthropic import Anthropic
from openai import OpenAI

from .config import Config

@dataclass
class TokenUsage:
    """Track token usage across models"""
    input: int = 0
    output: int = 0
    total_cost: float = 0.0
    by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)

class ResponseProcessor:
    @staticmethod
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

class LLMClients:
    def __init__(self):
        self.deepseek_client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.response_processor = ResponseProcessor()
        self.token_usage = TokenUsage()
        self.last_anthropic_call = 0  # Track last API call timestamp

    def _update_token_usage(self, model: str, tokens: Dict[str, int]) -> None:
        """Update token usage statistics"""
        self.token_usage.input += tokens["input"]
        self.token_usage.output += tokens["output"]
        
        if model not in self.token_usage.by_model:
            self.token_usage.by_model[model] = {"input": 0, "output": 0}
            
        self.token_usage.by_model[model]["input"] += tokens["input"]
        self.token_usage.by_model[model]["output"] += tokens["output"]

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting for Anthropic API calls"""
        elapsed = time.time() - self.last_anthropic_call
        if elapsed < Config.ANTHROPIC_REQUEST_DELAY:
            sleep_time = Config.ANTHROPIC_REQUEST_DELAY - elapsed
            time.sleep(sleep_time)
        self.last_anthropic_call = time.time()

    def get_deepseek_response(self, prompt: str, model_name: str) -> Tuple[str, str, Dict[str, int]]:
        """Get response from DeepSeek with retry logic and token tracking."""
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.deepseek_client.chat.completions.create(
                    model=model_name,  # Clearer parameter name
                    messages=[{"role": "user", "content": prompt}]
                )
                
                tokens = {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens
                }
                
                self._update_token_usage(model_name, tokens)  # Consistent naming
                
                return (
                    response.choices[0].message.reasoning_content or "",
                    response.choices[0].message.content or "",
                    tokens
                )
                
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise
                time.sleep(Config.RETRY_DELAY * (attempt + 1))  # Exponential backoff

    def get_claude_response(self, prompt: str, model: str) -> Tuple[str, Dict[str, int]]:
        """Get response from Claude with retry logic and token tracking."""
        self._enforce_rate_limit()  # Add rate limiting
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.anthropic_client.messages.create(
                    model=model,  # Use dynamic model name
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                tokens = {
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens
                }
                
                self._update_token_usage(model, tokens)  # Update token usage with dynamic model
                
                return (
                    response.content[0].text,
                    tokens
                )
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise
                time.sleep(Config.RETRY_DELAY * (attempt + 1))  # Exponential backoff

    def get_token_usage(self) -> TokenUsage:
        """Get current token usage statistics"""
        return self.token_usage
