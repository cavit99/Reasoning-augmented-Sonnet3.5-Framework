# ARC Challenge Model Benchmark

This repository contains a benchmarking tool for evaluating Large Language Models (LLMs) on the Abstraction and Reasoning Corpus (ARC) Challenge dataset. The benchmark currently supports DeepSeek Reasoner and Anthropic's Claude models, with a unique feature to test cross-model reasoning.

## Features

- Evaluates multiple LLMs on ARC Challenge questions
- Supports three evaluation modes:
  - DeepSeek Reasoner (deepseek-reasoner)
  - Claude 3.5 Sonnet (claude-3-5-sonnet-20241022)
  - Claude 3.5 Sonnet with DeepSeek Reasoner's reasoning (claude-3-5-sonnet-20241022)
- Generates detailed CSV reports and performance metrics
- Includes retry logic for API resilience
- Configurable sample size for testing

## Prerequisites

- Python 3.8+
- API keys for:
  - DeepSeek
  - Anthropic
  - Hugging Face (for dataset access)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arc-challenge-benchmark.git
cd arc-challenge-benchmark
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```env
DEEPSEEK_API_KEY=your_deepseek_key
ANTHROPIC_API_KEY=your_anthropic_key
HF_TOKEN=your_huggingface_token
```

## Usage

Run the benchmark:
```bash
python arc_challenge.py
```

To modify the number of samples or other configurations, edit the `Config` class in `arc_challenge.py`.

## Output

The benchmark generates two types of output files in the `benchmark_results` directory:

1. CSV file (`arc_results_[timestamp].csv`):
   - Detailed results for each question
   - Model predictions
   - Correctness indicators

2. JSON file (`arc_metrics_[timestamp].json`):
   - Overall accuracy metrics
   - Total samples processed
   - Timestamp

## Configuration

Key configurations in the `Config` class:

```python
MAX_SAMPLES: Optional[int] = 10  # Set to None for full dataset
DATASET_NAME: str = "ibragim-bad/arc_challenge"
DATASET_SPLIT: str = "test"  # Options: "test", "train", "validation"
```

### Dataset Splits
The ARC Challenge dataset used contains:
- Test: 1,119 examples (default for benchmarking)
- Train: 1,119 examples
- Validation: 299 examples

By default, the benchmark:
- Uses the test split for evaluation
- Randomly samples 10 questions (when MAX_SAMPLES is set)
- Uses a random seed for reproducible sampling

To run on the full test set, set `MAX_SAMPLES = None` in the Config class.

## Models

- **DeepSeek Reasoner**: Primary reasoning model
- **Claude 3.5 Sonnet**: Secondary model
- **Claude with Reasoning**: Claude augmented with DeepSeek's reasoning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- ARC Challenge dataset creators
- DeepSeek team
- Anthropic team

## Disclaimer

This tool is for research purposes. Please ensure you comply with all API providers' terms of service and have appropriate API access before using.