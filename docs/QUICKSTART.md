# Quick Start Guide

Get up and running with the Navigation Planning LLM research project in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning)
- API keys (optional, for API-based models)

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/navigation-planning-llm.git
cd navigation-planning-llm

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Generate Dataset

Generate the complete dataset with all format variants:

```bash
# Generate dataset (takes ~2-3 minutes)
python tools/generate_dataset.py --output_dir ./data_set --seed 42

# Expected output:
# [R1] Generated scene 1/10: ./data_set/R1/scene/scene_01.json
# [R1] Generated scene 2/10: ./data_set/R1/scene/scene_02.json
# ...
# Total queries generated: 1250
# Total samples (with variants): 8750
```

## Step 3: Validate Dataset

Verify dataset integrity and information equivalence:

```bash
# Run validation
python tools/validate_dataset.py --base_dir ./data_set --validate --check_equivalence

# Expected output:
# Total scenes: 25
# Total queries: 1250
# All scenes valid: ✓ YES
# Information equivalence: ✓ All variants encode same information
```

## Step 4: Explore Dataset

### View a Scene

```bash
# View a scene JSON file
cat data_set/R1/scene/scene_01.json | head -30
```

### Check Query Distribution

```bash
# Count queries by type
python -c "
import pandas as pd
import glob

all_queries = []
for f in glob.glob('data_set/*/queries/*.csv'):
    df = pd.read_csv(f)
    all_queries.append(df)

combined = pd.concat(all_queries)
print(combined['task_type'].value_counts())
"
```

## Step 5: Run Inference (Optional)

### Local Model (≤ 8B parameters)

```bash
# Download model first (if not already available)
# python tools/download_model.py --model qwen3-0.6b

# Run inference
bash run/run_local_qwen.sh
```

### API Model

```bash
# 1. Set API keys in run/run_api.py
# 2. Run inference
bash run/run_api.sh
```

## Step 6: Generate Figures (Optional)

Reproduce paper figures:

```bash
cd figures
python figure5.py
python figure6.py
python figure7.py
python figure8.py

# Output: figures saved in figures*/ directories
```

## Common Tasks

### Check Dataset Statistics

```bash
python tools/validate_dataset.py --base_dir ./data_set
```

### Re-generate with Different Seed

```bash
python tools/generate_dataset.py --output_dir ./data_set --seed 12345
```

### Add New Scenes

```python
# Edit tools/generate_dataset.py
# Modify generate_r1_scenes(), generate_r2_scenes(), or generate_c1_c2_scenes()
# Re-run generation
```

### Export to Different Format

```python
# Use tools/scene_serializer.py
from tools.scene_serializer import generate_all_variants
import json

with open('data_set/R1/scene/scene_01.json', 'r') as f:
    scene = json.load(f)

variants = generate_all_variants(scene['world'])
for name, text in variants.items():
    print(f"{name}: {len(text)} chars")
```

## Troubleshooting

### Issue: "Module not found"

**Solution**: Ensure virtual environment is activated and requirements are installed.

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Permission denied" on script execution

**Solution**: Make scripts executable.

```bash
chmod +x run/*.sh
chmod +x tools/*.py
```

### Issue: "API key not found"

**Solution**: Set API keys in `run/run_api.py` or use environment variables.

```bash
export OPENAI_API_KEY="your-key"
export QWEN_API_KEY="your-key"
```

### Issue: Dataset generation too slow

**Solution**: Reduce number of scenes for testing.

```bash
python tools/generate_dataset.py --r1_scenes 2 --r2_scenes 2 --conflict_scenes 1
```

## Next Steps

1. Read [README.md](README.md) for full documentation
2. Read [docs/DATASET.md](docs/DATASET.md) for dataset details
3. Read [docs/ENGINEERING_SUMMARY.md](docs/ENGINEERING_SUMMARY.md) for technical implementation
4. Run experiments with your models
5. Contribute improvements via pull requests

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: Open a GitHub issue
- **Questions**: Use GitHub Discussions

---

**Estimated Time to Complete**: 5-10 minutes for dataset generation, 1-2 hours for full experiment reproduction.
