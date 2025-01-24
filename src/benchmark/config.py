import json
import os
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    # Environment setup
    ENV_FILE = ".env"
    CONFIG_FILE = "config.json"
    
    # Application constants
    MAX_SAMPLES = 1  # Set to None to run on all samples
    DATASET_NAME = "iDavidRein/gpqa"
    DATASET_SPLIT = "train"  # Only split available in GPQA
    RESULTS_DIR = "benchmark_results"
    BATCH_SIZE = 100  # Number of samples per HuggingFace upload
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    # Model configuration
    DEEPSEEK_MODEL = "deepseek-reasoner"
    CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
    
    # Dataset configuration
    HF_DATASET_REPO = "spawn99/GPQA-diamond-ClaudeR1"
    
    @classmethod
    def _validate_config_json(cls) -> Dict[str, Any]:
        """Validate config.json structure and content"""
        if not os.path.exists(cls.CONFIG_FILE):
            raise FileNotFoundError("Missing config.json file with model specifications")
            
        try:
            with open(cls.CONFIG_FILE) as f:
                config = json.load(f)
                
            if "models" not in config:
                raise ValueError("Invalid config.json: missing 'models' key")
                
            required_model_keys = {"model_key", "display_name", "token_groups", "tokens", "cost_per_million"}
            required_group_keys = {"type", "input", "output"}
            required_cost_keys = {"input", "output"}
            
            for model in config["models"]:
                missing_keys = required_model_keys - set(model.keys())
                if missing_keys:
                    raise ValueError(f"Model missing required keys: {missing_keys}")
                    
                for group in model["token_groups"]:
                    missing_keys = required_group_keys - set(group.keys())
                    if missing_keys:
                        raise ValueError(f"Token group missing required keys: {missing_keys}")
                        
                missing_cost_keys = required_cost_keys - set(model["cost_per_million"].keys())
                if missing_cost_keys:
                    raise ValueError(f"Cost per million missing required keys: {missing_cost_keys}")
                    
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config.json: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error validating config.json: {str(e)}")
    
    @classmethod
    def validate(cls) -> None:
        """Validate environment setup and create necessary directories"""
        # Validate .env file
        if not os.path.exists(cls.ENV_FILE):
            raise FileNotFoundError(
                f"Missing {cls.ENV_FILE} file. Please create one with DEEPSEEK_API_KEY, "
                "ANTHROPIC_API_KEY, and HF_TOKEN."
            )
        
        load_dotenv(dotenv_path=cls.ENV_FILE, override=True)
        
        # Validate API keys
        required_keys = ['DEEPSEEK_API_KEY', 'ANTHROPIC_API_KEY', 'HF_TOKEN']
        missing = [k for k in required_keys if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing API keys: {', '.join(missing)}")
        
        # Validate config.json
        cls._validate_config_json()
        
        # Create necessary directories
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
    
    @staticmethod
    def get_output_path() -> str:
        """Generate standardized output path for results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"arc__deepseek-{Config.DEEPSEEK_MODEL}__claude-{Config.CLAUDE_MODEL}__{timestamp}"
        return os.path.join(Config.RESULTS_DIR, f"{base_name}.jsonl")
