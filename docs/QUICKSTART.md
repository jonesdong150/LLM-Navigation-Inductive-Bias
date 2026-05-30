# Quick Start Guide

Get up and running in 5 minutes.

## Prerequisites

- Python 3.10+
- pip

## One-Click Setup

```bash
git clone https://github.com/yourusername/navigation-planning-llm.git
cd navigation-planning-llm
bash setup.sh --full
```

This will: create a virtual environment, install dependencies, generate the dataset, run validation, and execute all unit tests.

## Manual Setup

### Step 1: Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Step 2: Generate Dataset

```bash
python tools/generate_dataset.py --output_dir ./data_set --seed 42
```

Output:
```
[R1] Generating simple navigation scenes...
  [R1] scene_01: 18 variants, 50 queries
  ...
Total scenes: 30
Total queries: 1500
```

### Step 3: Validate

```bash
python tools/validate_dataset.py --base_dir ./data_set --validate --check_equivalence
```

Expected: All 30 scenes valid, all 1,500 queries valid, information equivalence verified.

### Step 4: Run Tests

```bash
python tests/test_scene_serializer.py
python tests/test_generate_dataset.py
```

## Using Makefile

```bash
make setup          # Create venv + install deps
make generate       # Generate dataset
make validate       # Validate dataset
make test           # Run unit tests
make verify         # Verify query ground truths
make all            # Full pipeline
make clean          # Remove generated files
make help           # Show all commands
```

## Dataset Structure

```
data_set/
├── R1/                          # Simple navigation (4-6 rooms)
│   ├── scene/scene_*.json       # Scene JSON with world + variants
│   └── queries/*_questions.csv  # 50 queries per scene
├── R2/                          # Complex navigation (6-14 rooms)
│   ├── scene/
│   └── queries/
└── conflict/                    # Conflict test scenes
    ├── scene/
    └── queries/
```

Each scene JSON contains:
- `world`: Structured ground truth (rooms, edges, objects, history)
- `variants`: All format variants (flat, hier, clustered x 100%/50%/25%)

Each query CSV contains:
- `question_id`, `task_type`, `question_text`, `answer_format`, `ground_truth`

## Running Inference

### Local Model

```bash
# Qwen models (requires GPU)
bash run/run_local_qwen.sh --models Qwen3-4B --scenes R1

# Llama models (requires GPU)
bash run/run_local_llama.sh --models Llama-3.2-3B --scenes R1
```

### API Model

```bash
# Set API keys
export QWEN_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run
bash run/run_api.sh
```

## Troubleshooting

| Issue | Solution |
|:---|:---|
| `Module not found` | Activate venv: `source venv/bin/activate` |
| `Permission denied` | `chmod +x run/*.sh setup.sh` |
| `API key not found` | Set env vars or edit `run/run_api.py` |
| Generation too slow | Use `--r1_scenes 2 --r2_scenes 2` for testing |

## Next Steps

- [Dataset Details](DATASET.md) - Format variants, query types, schema
- [Engineering Summary](ENGINEERING_SUMMARY.md) - Architecture and design
- [README.md](../README.md) - Full documentation
