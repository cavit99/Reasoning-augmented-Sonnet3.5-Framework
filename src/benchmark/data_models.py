from __future__ import annotations  # Enable modern type hint syntax
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Type
import json
import os

@dataclass
class SchemaField:
    """Combines type validation and serialization format"""
    py_type: Type[Any]  # Use Type for type objects
    hf_type: str | None = None  # Use union operator

    def __post_init__(self):
        if self.hf_type is None:
            type_map: Dict[Type[Any], str] = {
                str: "string",
                int: "int64",
                float: "float64",
                bool: "bool",
                type(None): "null"
            }
            self.hf_type = type_map.get(self.py_type, "string")

class BaseDataset(ABC):
    @abstractmethod
    def get_formatted_prompt(self, problem: Dict[str, Any]) -> str:
        pass

class GPQADataset(BaseDataset):
    def get_formatted_prompt(self, problem: Dict[str, Any]) -> str:
        """Format the prompt using the Question field directly"""
        return (
            f"{problem['Question']}\n\n"
            "Please provide a detailed explanation of your reasoning, "
            "then state your final answer clearly enclosed in <answer>...</answer> XML tags."
        )

class SchemaManager:
    @staticmethod
    def _get_config() -> Dict[str, Any]:
        """Load model config from JSON file"""
        try:
            # Get the root directory by going up two levels from the current file
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(root_dir, "config.json")
            
            with open(config_path) as f:
                config = json.load(f)
                if "models" not in config:
                    raise ValueError("Invalid config.json: missing 'models' key")
                return config["models"]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config.json: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading config.json: {str(e)}")

    @staticmethod
    def get_record_schema() -> Dict[str, Union[SchemaField, Dict[str, Any]]]:
        models = SchemaManager._get_config()
        schema: Dict[str, Any] = {
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
            },
            "costs": { 
                "type": "nested",
                "structure": {
                    "total": SchemaField(float),
                    **{
                        f"{model['model_key']}_{group['label'].lower().replace(' ', '_')}"
                        if group["type"] == "grouped" else model['model_key']: SchemaField(float)
                        for model in models 
                        for group in model["token_groups"]
                    }
                }
            }
        }

        # Add top-level answer and grade fields for all models/groups
        for model in models:
            for group in model["token_groups"]:
                if group["type"] == "grouped":
                    key_suffix = f"{model['model_key']}_{group['label'].lower().replace(' ', '_')}"
                    schema[f"{key_suffix}_answer"] = SchemaField(str)
                    schema[f"{key_suffix}_grade"] = SchemaField(type(None))
                else:
                    schema[f"{model['model_key']}_answer"] = SchemaField(str)
                    schema[f"{model['model_key']}_grade"] = SchemaField(type(None))
        
        return schema

    @staticmethod
    def get_model_response_structure(model_config: Dict[str, Any]) -> Dict[str, Any]:
        structure: Dict[str, Any] = {}
        for group in model_config["token_groups"]:
            if group["type"] == "grouped":
                key = group["label"].lower().replace(" ", "_")
                structure[key] = {
                    "full_response": SchemaField(str)
                }
            else:
                structure.update({
                    "full_response": SchemaField(str),
                    "reasoning": SchemaField(str)  
                })
        return structure

    @staticmethod
    def get_token_usage_structure() -> Dict[str, Dict[str, SchemaField]]:
        structure = {}
        for model in SchemaManager._get_config():
            model_key = model["model_key"]
            for group in model["token_groups"]:
                if group["type"] == "grouped":
                    group_key = f"{model_key}_{group['label'].lower().replace(' ', '_')}"
                    structure[group_key] = {
                        "input": SchemaField(int),
                        "output": SchemaField(int)
                    }
                else:
                    structure[model_key] = {
                        "input": SchemaField(int),
                        "output": SchemaField(int)
                    }
        return structure

    @staticmethod
    def validate_record(record: Dict[str, Any]) -> bool:
        """Validate record structure using central schema."""
        schema = SchemaManager.get_record_schema()
        
        def check_structure(data: Any, schema_node: Union[SchemaField, Dict[str, Any]]) -> bool:
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
        
        # Check all top-level fields
        for key, schema_field in schema.items():
            if key not in ["model_responses", "metadata", "token_usage", "costs"]:
                if not check_structure(record.get(key), schema_field):
                    return False
        
        # Check nested structures
        return check_structure(record.get("model_responses"), schema["model_responses"]) and \
               check_structure(record.get("metadata"), schema["metadata"]) and \
               check_structure(record.get("token_usage"), schema["token_usage"]) and \
               check_structure(record.get("costs"), schema["costs"])
