# The Sword, Shield, and Achilles' Heel: Characterizing the Linguistic Inductive Bias of LLMs for Spatial Reasoning in Navigation Planning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/dataset-30%20scenes%20%7C%201%2C500%20queries-green.svg)](docs/DATASET.md)

> Official implementation of the paper: **"The Sword, Shield, and Achilles' Heel: Characterizing the Linguistic Inductive Bias of Large Language Models for Spatial Reasoning in Navigation Planning"**

## Abstract

Large Language Models (LLMs) are increasingly used for spatial reasoning in navigation planning, yet the extent to which their performance depends on linguistic surface forms rather than genuine spatial understanding remains poorly characterized. We introduce a **dual-interventional framework** that disentangles **representation intervention** (how spatial information is formatted) from **context intervention** (how semantic labels are varied or corrupted), enabling systematic diagnosis of LLM spatial reasoning capabilities.

Our dataset generation pipeline is **fully automated, rule-based, and deterministic** -- no human annotation or LLM-based annotation is used. All ground truths are verified by symbolic solvers (BFS, coordinate comparison, direct lookup) with 100% accuracy.

## Key Contributions

| Contribution | Description |
|:---|:---|
| **Knowledge Base** | Formal KB with 12 room categories, 10 object categories, 8 room/object attributes, each with canonical name, synonyms, and abbreviation |
| **Representation Intervention** | 3 format styles (Flat, Hierarchical, Clustered) x 3 compression levels (100%, 50%, 25%) = 9 structural variants per scene |
| **Context Intervention** | Semantic Variation (synonym substitution) + Semantic Conflict (duplicate labels for distinct nodes) |
| **Dimension Ablation** | Selective removal of spatial dimensions: Topology-only, Geometry-only, Semantics-only |
| **Symbolic Verification** | All 1,500 query ground truths verified by symbolic solvers at 100% accuracy |
| **Dual-LLM Verification** | Framework for Generator LLM != Verifier LLM cross-validation |

## Repository Structure

```
.
├── tools/                        # Core pipeline
│   ├── knowledge_base.py         # Room/object/attribute knowledge base
│   ├── build_scene.py            # Rule-based world generator
│   ├── scene_serializer.py       # Text serialization (all format variants)
│   ├── generate_dataset.py       # Dataset generation pipeline
│   ├── validate_dataset.py       # Validation and statistics
│   ├── verify_queries.py         # Symbolic + dual-LLM verification
│   ├── build_queries.py          # PathGen query builder
│   └── aggregate_results.py      # Results aggregation
├── tests/                        # Unit tests
│   ├── test_scene_serializer.py  # 12 tests for serializer
│   └── test_generate_dataset.py  # 9 tests for dataset pipeline
├── run/                          # Inference scripts
│   ├── run_api.py / run_api.sh   # API-based inference (Qwen, GPT, Gemini)
│   ├── run_local_qwen.py / .sh   # Local Qwen inference
│   └── run_local_llama.py / .sh  # Local Llama inference
├── eval/                         # Evaluation results (per-model summaries)
├── prompts/                      # Prompt templates (L1-L4 strictness levels)
├── data_set/                     # Generated dataset (R1, R2, conflict)
├── docs/                         # Documentation
├── Makefile                      # One-click commands
├── setup.sh                      # One-click environment setup
└── requirements.txt              # Python dependencies
```

## Quick Start

### One-Click Setup

```bash
git clone https://github.com/yourusername/navigation-planning-llm.git
cd navigation-planning-llm
bash setup.sh          # Create venv, install deps, generate dataset, run tests
```

### Manual Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python tools/generate_dataset.py --output_dir ./data_set --seed 42

# 4. Validate dataset
python tools/validate_dataset.py --base_dir ./data_set --validate --check_equivalence

# 5. Run tests
python tests/test_scene_serializer.py
python tests/test_generate_dataset.py
```

### Using Makefile

```bash
make setup          # Create venv + install deps
make generate       # Generate dataset
make validate       # Validate dataset
make test           # Run unit tests
make all            # setup + generate + validate + test
make clean          # Remove generated files
```

## Methodology

### Dual-Interventional Framework

```
                    ┌─────────────────────────────────┐
                    │     Knowledge Base (KB)          │
                    │  Rooms, Objects, Attributes,     │
                    │  Synonyms, Abbreviations         │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │  Scene     │   │  Context   │   │  Query     │
     │  Generation│   │  Interven. │   │  Generation│
     │  (Rule-    │   │  (Semantic │   │  (Symbolic │
     │   based)   │   │   Var/     │   │   Solvers) │
     │            │   │   Conflict)│   │            │
     └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
           │                │                │
           ▼                ▼                ▼
     ┌─────────────────────────────────────────────┐
     │            Format Variants                    │
     │  Flat / Hierarchical / Clustered             │
     │  x 100% / 50% / 25% retention               │
     └──────────────────────┬──────────────────────┘
                            │
                            ▼
     ┌─────────────────────────────────────────────┐
     │         Information Equivalence Check         │
     │  All variants encode identical spatial truths │
     └─────────────────────────────────────────────┘
```

### Representation Intervention (Format Variants)

Each scene is serialized into **9 structural variants**:

| Format | 100% | 50% | 25% |
|:---|:---|:---|:---|
| **Flat** | Full narrative | Reduced fillers | Abbreviations only |
| **Hierarchical** | `[R1: Library @ (0,0)]` | Semi-structured | Maximum compression |
| **Clustered** | `Zone A: {R1,R2}` | Compressed zones | Numeric indices |

### Context Intervention

| Type | Method | Example |
|:---|:---|:---|
| **Semantic Variation** | Synonym substitution from KB | "Corridor" -> "Hallway", "Office" -> "Workspace" |
| **Semantic Conflict** | Duplicate labels for distinct nodes | R2="Office", R5="Lab", but both labeled "Office" |

### Query Types

| Type | Question | Answer | Solver |
|:---|:---|:---|:---|
| **ObjectLocation** | Where is the Sofa? | R3 | Direct lookup |
| **GeometryYN** | Is R1 North of R2? | YES/NO | Coordinate comparison |
| **TopologyYN** | Are R1 and R3 connected? | YES/NO | Edge lookup |
| **ReachabilityYN** | Can you reach R4 from R1? | YES/NO | BFS |
| **PathGen** | Shortest path from R1 to R4? | R1->R2->R4 | BFS shortest path |

## Dataset Statistics

| Dataset | Scenes | Rooms | Queries | Variants | Total Samples |
|:---|:---:|:---:|:---:|:---:|:---:|
| **R1** (Simple) | 10 | 4-6 | 500 | 18 (9 structural + 9 semantic) | 9,000 |
| **R2** (Complex) | 10 | 6-14 | 500 | 22 (4 ablation + 9 structural + 9 semantic) | 11,000 |
| **Conflict** | 10 | 6-14 | 500 | 7 (4 base + 3 conflict) | 3,500 |
| **Total** | **30** | | **1,500** | | **23,500** |

### Complexity Gradients (R2)

| Gradient | Rooms | Room Types | Objects/Room | History Steps |
|:---|:---:|:---:|:---:|:---:|
| G1 | 6 | 3 | 4 | 2 |
| G2 | 8 | 4 | 6 | 4 |
| G3 | 10 | 5 | 8 | 6 |
| G4 | 12 | 6 | 10 | 8 |
| G5 | 14 | 7 | 12 | 10 |

## Reproducibility

The dataset generation is **completely deterministic**:

```bash
# Same seed always produces identical dataset
python tools/generate_dataset.py --seed 42

# Verify determinism
python tools/generate_dataset.py --seed 42 --output_dir ./data_set_v1
python tools/generate_dataset.py --seed 42 --output_dir ./data_set_v2
diff -r ./data_set_v1 ./data_set_v2   # Should produce no differences
```

### Validation

```bash
# Full validation: scene integrity + query integrity + information equivalence
python tools/validate_dataset.py --base_dir ./data_set --validate --check_equivalence --verbose

# Symbolic verification of query ground truths
python tools/verify_queries.py data_set/R1/scene/scene_01.json

# Unit tests
python tests/test_scene_serializer.py
python tests/test_generate_dataset.py
```

## Running Experiments

### Local Model (<= 8B parameters)

```bash
# Qwen models
bash run/run_local_qwen.sh --models Qwen3-4B --scenes R1

# Llama models
bash run/run_local_llama.sh --models Llama-3.2-3B --scenes R1
```

### API-based Model

```bash
# Set API keys in environment
export QWEN_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run inference
bash run/run_api.sh
```

### Aggregating Results

```bash
python tools/aggregate_results.py
```

## Documentation

| Document | Description |
|:---|:---|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 5-minute quick start guide |
| [docs/DATASET.md](docs/DATASET.md) | Dataset structure and format details |
| [docs/ENGINEERING_SUMMARY.md](docs/ENGINEERING_SUMMARY.md) | Technical implementation summary |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2026sword,
      title={The Sword, Shield, and Achilles' Heel: Characterizing the Linguistic Inductive Bias of Large Language Models for Spatial Reasoning in Navigation Planning},
      author={Xudong Zhang and Jian Yang and Shengkai Wang and Jiangpeng Tian and Shaowen Chen and Xian Wei and Ke Li and Xiong You},
      journal={arXiv preprint arXiv:2605.31404},
      year={2026},
      eprint={2605.31404},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.31404},
      doi={10.48550/arXiv.2605.31404}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## Contact

For questions and feedback, please open an issue on GitHub.

---

**Note**: This repository is released for academic research purposes. The dataset generation is fully automated, rule-based, and deterministic -- no human annotation or LLM-based annotation is used. All ground truths are verified by symbolic solvers with 100% accuracy.
