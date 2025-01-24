import json
import os
import time
from typing import Optional, Set, List, Dict, Any

from datasets import load_dataset, Features, Value, Dataset, concatenate_datasets

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

    def save_result(self, result: Dict[str, Any], output_path: str) -> None:
        """Save a single result to the output file."""
        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + '\n')

    def upload_to_huggingface(self, jsonl_path: str, num_records: int) -> None:
        """Upload results with dynamic schema generation."""
        def build_features(schema_node):
            if isinstance(schema_node, SchemaField):
                return Value(schema_node.hf_type)
            if isinstance(schema_node, dict):
                if "type" in schema_node and schema_node["type"] == "nested":
                    return {k: build_features(v) for k, v in schema_node["structure"].items()}
                return {k: build_features(v) for k, v in schema_node.items()}
            return schema_node
        
        schema = self.schema_manager.get_record_schema()
        features = Features(build_features(schema))
        
        # Load existing dataset with retry
        existing_ds = None
        for attempt in range(Config.MAX_RETRIES):
            try:
                existing_ds = load_dataset(Config.HF_DATASET_REPO, split="train")
                break
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    print(f"Warning: Could not load existing dataset: {str(e)}")
                time.sleep(Config.RETRY_DELAY * (attempt + 1))

        # Load and upload new dataset with retry
        for attempt in range(Config.MAX_RETRIES):
            try:
                new_ds = load_dataset('json', data_files=jsonl_path, features=features, split='train')
                combined_ds = concatenate_datasets([existing_ds, new_ds]) if existing_ds else new_ds
                
                combined_ds.push_to_hub(
                    repo_id=Config.HF_DATASET_REPO,
                    token=os.getenv('HF_TOKEN'),
                    private=False,
                    commit_message=f"Add batch of {num_records} results"
                )
                print(f"✅ Uploaded {num_records} new records to {Config.HF_DATASET_REPO}")
                break
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    print(f"❌ Upload failed: {str(e)}")
                    raise
                print(f"Retrying upload after error: {str(e)}")
                time.sleep(Config.RETRY_DELAY * (attempt + 1))

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and prepare the dataset for processing."""
        processed_ids = self.get_processed_record_ids()
        print(f"Found {len(processed_ids)} processed records")

        # Load and filter dataset with retry
        for attempt in range(Config.MAX_RETRIES):
            try:
                dataset = load_dataset(Config.DATASET_NAME, "gpqa_diamond", token=os.getenv('HF_TOKEN'))
                data = [d for d in dataset[Config.DATASET_SPLIT] if d['Record ID'] not in processed_ids]
                print(f"Found {len(data)} unprocessed records")
                return data
            except Exception as e:
                if attempt == Config.MAX_RETRIES - 1:
                    raise ValueError(f"Failed to load dataset after {Config.MAX_RETRIES} attempts: {str(e)}")
                print(f"Retrying dataset load after error: {str(e)}")
                time.sleep(Config.RETRY_DELAY * (attempt + 1))
