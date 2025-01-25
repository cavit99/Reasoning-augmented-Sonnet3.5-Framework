import json
import argparse
from collections import defaultdict, OrderedDict

# Define the required field order
FIELD_ORDER = [
    "record_id",
    "question",
    "correct_answer",
    "correct_explanation",
    "deepseek_reasoner_answer",
    "deepseek_reasoner_grade",
    "deepseek_reasoner_eval",
    "claude_sonnet_standalone_answer",
    "claude_sonnet_standalone_grade",
    "claude_sonnet_standalone_eval",
    "claude_sonnet_with_reasoning_answer",
    "claude_sonnet_with_reasoning_grade",
    "claude_sonnet_with_reasoning_eval",
    "metadata",
    "token_usage",
    "costs",
    "model_responses"
]

def load_records(file_path):
    """Load JSONL records with flexible ID handling"""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
    return records

def create_lookup(records):
    """Create multiple lookup indexes for record matching"""
    lookup = {
        'by_id': defaultdict(list),
        'by_question': defaultdict(list),
        'by_answer': defaultdict(list)
    }
    
    for record in records:
        if 'record_id' in record:
            lookup['by_id'][record['record_id']].append(record)
        
        question = record.get('question', '').strip()[:100]
        if question:
            lookup['by_question'][question].append(record)
            
        answer = record.get('correct_answer', '').strip()
        if answer:
            lookup['by_answer'][answer].append(record)
            
    return lookup

def enforce_field_order(merged_record):
    """Enforce the specific field ordering"""
    ordered_record = OrderedDict()
    
    for field in FIELD_ORDER:
        ordered_record[field] = merged_record.get(field)
    
    for key in merged_record:
        if key not in ordered_record:
            ordered_record[key] = merged_record[key]
    
    return ordered_record

def merge_records(record_a, record_b):
    """Merge two records with priority to non-null values"""
    merged = {}
    all_keys = set(record_a.keys()) | set(record_b.keys())
    
    for key in all_keys:
        val_a = record_a.get(key)
        val_b = record_b.get(key)
        
        if val_b is not None:
            merged[key] = val_b
        else:
            merged[key] = val_a
            
    return enforce_field_order(merged)

def consolidate_files(file1, file2, output_file):
    """Main consolidation function with proper variable handling"""
    records1 = load_records(file1)
    records2 = load_records(file2)
    
    lookup1 = create_lookup(records1)
    lookup2 = create_lookup(records2)
    
    matched = set()
    consolidated = []
    
    # Match by record_id
    for record_id in lookup1['by_id']:
        if record_id in lookup2['by_id']:
            for rec1 in lookup1['by_id'][record_id]:
                for rec2 in lookup2['by_id'][record_id]:
                    merged = merge_records(rec1, rec2)
                    consolidated.append(merged)
                    matched.add(id(rec1))
                    matched.add(id(rec2))
    
    # Match by question
    for question in lookup1['by_question']:
        if question in lookup2['by_question']:
            for rec1 in lookup1['by_question'][question]:
                for rec2 in lookup2['by_question'][question]:
                    if id(rec1) not in matched and id(rec2) not in matched:
                        merged = merge_records(rec1, rec2)
                        consolidated.append(merged)
                        matched.add(id(rec1))
                        matched.add(id(rec2))
    
    # Match by answer
    for answer in lookup1['by_answer']:
        if answer in lookup2['by_answer']:
            for rec1 in lookup1['by_answer'][answer]:
                for rec2 in lookup2['by_answer'][answer]:
                    if id(rec1) not in matched and id(rec2) not in matched:
                        merged = merge_records(rec1, rec2)
                        consolidated.append(merged)
                        matched.add(id(rec1))
                        matched.add(id(rec2))
    
    # Add remaining records
    for rec in records1 + records2:
        if id(rec) not in matched:
            ordered_rec = enforce_field_order(rec)
            consolidated.append(ordered_rec)
    
    # Write output
    with open(output_file, 'w') as f:
        for record in consolidated:
            f.write(json.dumps(dict(record), ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consolidate JSONL files with field ordering')
    parser.add_argument('file1', help='First input file')
    parser.add_argument('file2', help='Second input file')
    parser.add_argument('output', help='Output file')
    args = parser.parse_args()
    
    consolidate_files(args.file1, args.file2, args.output)