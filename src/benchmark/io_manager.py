import json
import os
from typing import Optional, Set, List, Dict, Any

from datasets import load_dataset
from .cost_manager import CostManager
from .config import Config
from .data_models import SchemaField, SchemaManager

class IOManager:
    def __init__(self):
        self.schema_manager = SchemaManager()
        
    def get_processed_record_ids(self) -> Set[str]:
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

    def consolidate_jsonl_files(self) -> Optional[str]:
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
                            if not self.schema_manager.validate_record(record):
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
    
    def load_existing_results(self) -> Dict[str, Dict[str, Any]]:
        """Load all existing results into a dictionary by record_id"""
        existing = {}
        consolidated_path = os.path.join(Config.RESULTS_DIR, "consolidated.jsonl")
        
        if os.path.exists(consolidated_path):
            with open(consolidated_path) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        existing[record["record_id"]] = record
                    except json.JSONDecodeError:
                        continue
        return existing

    def save_result(self, result: Dict[str, Any], output_path: str) -> None:
        """Save a single result to the output file."""
        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')

    def load_dataset(self) -> tuple[list, list]:
        """Returns (existing_records_to_update, new_records_to_process)"""
        processed_ids = self.get_processed_record_ids()
        existing_records = self.load_existing_records()  # New method
        
        # Load fresh dataset
        dataset = load_dataset(Config.DATASET_NAME, "gpqa_diamond", token=os.getenv('HF_TOKEN'))
        all_data = dataset[Config.DATASET_SPLIT]
        
        # Separate existing and new
        existing_to_update = []
        new_to_process = []
        
        for record in all_data:
            record_id = str(record['Record ID'])
            if record_id in processed_ids:
                # Check which models are missing
                existing = existing_records.get(record_id, {})
                missing_models = self._get_missing_models(existing)
                if missing_models:
                    existing_to_update.append({
                        'base_data': record,
                        'existing': existing,
                        'missing_models': missing_models
                    })
            else:
                new_to_process.append(record)
        
        return existing_to_update, new_to_process

    def _get_missing_models(self, existing_record: dict) -> list:
        """Identify models not present in existing record"""
        expected_models = {m['model_key'] for m in CostManager().model_config}
        present_models = set()
        
        for key in existing_record.keys():
            if '_answer' in key:
                model = key.split('_')[0]
                present_models.add(model)
        
        return list(expected_models - present_models)

    def save_updated_record(self, updated_record: dict, output_path: str) -> None:
        """Update existing record in-place"""
        temp_path = output_path + ".tmp"
        
        with open(output_path, 'r') as old_file, open(temp_path, 'w') as new_file:
            for line in old_file:
                record = json.loads(line)
                if record['record_id'] == updated_record['record_id']:
                    # Merge updates
                    merged = {**record, **updated_record}
                    new_file.write(json.dumps(merged) + '\n')
                else:
                    new_file.write(line)
        
        os.replace(temp_path, output_path)