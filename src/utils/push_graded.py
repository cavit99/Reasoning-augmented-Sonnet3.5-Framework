# push_graded.py
import os
from datasets import load_dataset, Dataset
from dotenv import load_dotenv

load_dotenv() 

def push_graded():
    # Use absolute path resolution from project root
    graded_path = ""

    try:
        # Load graded data
        ds = load_dataset("json", data_files=graded_path, split="train")
        
        # Push to hub
        ds.push_to_hub(
            "spawn99/GPQA-diamond-ClaudeR1",  
            token=os.getenv("HF_TOKEN"),
            private=False,
            commit_message="Update with graded results"
        )
        print("✅ Successfully pushed graded data to hub")
    except Exception as e:
        print(f"❌ Error during dataset push: {str(e)}")
        return

if __name__ == "__main__":
    push_graded()