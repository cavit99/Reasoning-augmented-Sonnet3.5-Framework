import json
import os
from google import genai
from google.genai import types
import time
from typing import List, Dict, Any
from tqdm import tqdm


# Get the absolute path to the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Create benchmark_results directory path relative to the script location
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../benchmark_results"))
os.makedirs(RESULTS_DIR, exist_ok=True)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

grading_config = types.GenerateContentConfig(
    temperature=0.3, 
    top_p=0.95,
    top_k=40,
    max_output_tokens=2048,
    response_mime_type="application/json",
    response_schema={
        "type": "object",
        "properties": {
            "deepseek": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string"
                    },
                    "success": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "reasoning",
                    "success"
                ]
            },
            "claude_standalone": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string"
                    },
                    "success": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "reasoning",
                    "success"
                ]
            },
            "claude_reasoning": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string"
                    },
                    "success": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "reasoning",
                    "success"
                ]
            }
        },
        "required": [
            "deepseek",
            "claude_standalone",
            "claude_reasoning"
        ]
    }
)

def grade_answers(record):
    prompt = f"""You are an expert scientific evaluator tasked with assessing the accuracy of AI-generated answers to questions against a provided ground truth. Your expertise spans various scientific fields including astrophysics, chemistry, physics, quantum mechanics, and biology.

    First, review the evaluation rules:

    <rules>
    1. Numerical equivalence: 3.14 ≈ ~3.14
    2. Notation equivalence: "propanoate" = "propionate"
    3. If answer surrounded by irrelevant characters ignore them. Accept answer if itself is correct.
    </rules>

    Now, examine the following information:
    <question>{record['question']}</question>
    <ground_truth>{record['correct_answer']}</ground_truth>

    <ground_truth_explanation>{record['correct_explanation']}</ground_truth_explanation>

    You will evaluate three different AI-generated answers against this ground truth:

    <deepseek_answer>{record['deepseek_reasoner_answer']}</deepseek_answer>

    <claude_standalone_answer>{record['claude_sonnet_standalone_answer']}</claude_standalone_answer>

    <claude_reasoning_answer>{record['claude_sonnet_with_reasoning_answer']}</claude_reasoning_answer>

    Your task is to evaluate each answer's accuracy. You must do reasoning by following these steps for each AI-generated answer:

    1. Carefully read and understand the ground truth and its explanation.
    2. For each AI-generated answer, compare it to the ground truth.
    3. Determine whether each answer successfully matches the ground truth.
    4. Consider arguments for classifying the answer as successful.
    5. Consider arguments against classifying the answer as successful.
    6. Make a final determination based on the evidence and arguments presented.
    7. Completeness of the answer in addressing all aspects of the ground truth.
    8. Any nuances or subtleties in the explanations that might affect the evaluation.

    After this thorough reasoning, for each answer make a final boolean success assessment in JSON format.
    Remember do not attempt to answer the <question> itself, you are only evaluating the answers against the ground truth."""
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=grading_config
    )
    time.sleep(6.667)
    for attempt in range(3):
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            if attempt < 2:  # retry if not the last attempt
                print(f"Attempt {attempt+1} failed: Invalid JSON response. Retrying...")
                time.sleep(1)  # wait a bit before retrying
            else:
                print(f"Attempt {attempt+1} failed: Invalid JSON response. Giving up.")  
                raise ValueError("Gemini returned invalid JSON") from e

# Use os.path.join for reliable path handling
temp_jsonl_path = os.path.join(RESULTS_DIR, "/Users/caviterginsoy/Library/CloudStorage/Dropbox/Coding/R1-reasoning-extraction-arc-challenge/benchmark_results/extracted_fields.jsonl")
temp_graded_path = os.path.join(RESULTS_DIR, "temp_graded.jsonl")

# Better approach: Open output file once, using 'w' mode to start fresh
with open(temp_jsonl_path) as f, open(temp_graded_path, "w") as outfile:
    # Read all lines first to get accurate progress
    lines = list(f)
    
    # Add tqdm progress bar with description
    for line in tqdm(lines, desc="Grading answers", unit="record"):
        record = json.loads(line)
        
        # Add this block to enforce schema requirements
        if "metadata" not in record:
            record["metadata"] = {}
        record["metadata"].setdefault("difficulty", "Unspecified")
        
        grades = grade_answers(record)
        
        # Update record with grades (existing code remains the same)
        record["deepseek_reasoner_grade"] = grades["deepseek"]["success"]
        record["deepseek_reasoner_eval"] = grades["deepseek"]["reasoning"]
        record["claude_sonnet_standalone_grade"] = grades["claude_standalone"]["success"]
        record["claude_sonnet_standalone_eval"] = grades["claude_standalone"]["reasoning"]
        record["claude_sonnet_with_reasoning_grade"] = grades["claude_reasoning"]["success"]
        record["claude_sonnet_with_reasoning_eval"] = grades["claude_reasoning"]["reasoning"]
        
        # Write to file
        json.dump(record, outfile)
        outfile.write("\n")