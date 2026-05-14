# Engineering Summary

## Project Overview

This document summarizes the engineering work completed to make this research project fully reproducible and compliant with academic community standards.

## Architecture

### Three-Layer Design

```
Layer 1: World Generation (tools/build_scene.py)
  - Rule-based generation of room layouts, topology, objects, history
  - G1-G5 complexity gradients (6-14 rooms)
  - Deterministic, seed-controlled

Layer 2: Text Serialization (tools/scene_serializer.py)
  - Structural compression: flat, hierarchical (0/50/75%), clustered (0/50/75%)
  - Dimension ablation: full, topo+hist, geom+rule+hist, sem+rule+hist
  - Conflict injection: topo-conflict, geom-conflict, sem-conflict
  - Information equivalence verification

Layer 3: Dataset Pipeline (tools/generate_dataset.py)
  - R1: Simple scenes → structural compression variants
  - R2: Complex scenes → dimension-ablation + structural variants
  - Conflict: Conflict scenes → base + conflict variants
  - Query generation: 5 task types, 50 queries per scene
```

### Information Flow

```
generate_world(seed) → world dict {rooms, edges, objects, history, rules}
    ↓
generate_all_variants(world) → {flat, hier, hier_50, ..., clustered_25}
generate_dimension_variants(world) → {flat_full, flat_topo_hist, ...}
generate_conflict_variants(world, seed) → {..., conflict_topo, conflict_geom, conflict_sem}
    ↓
generate_queries_for_scene(scene) → [50 queries with ground truth]
    ↓
scene JSON + questions CSV per scene
```

## Key Design Decisions

### 1. Unified Serialization

All serialization methods live in a single module (`tools/scene_serializer.py`) rather than spread across multiple files. This ensures:
- Consistent information equivalence checking
- No duplicate code between experiment types
- Single source of truth for format definitions

### 2. Separation of World and Rendering

World generation (`tools/build_scene.py`) is separate from text rendering (`tools/scene_serializer.py`). This allows:
- Independent testing of world generation vs serialization
- Different experiment types can use the same world with different serialization strategies
- Clean extension with new variant types

### 3. Numeric Index Fallback for Compressed Formats

Compressed formats (hier_25, clustered_25) use numeric indices (1, 2, 3) instead of room IDs (R1, R2, R3). The equivalence checker handles both representations.

## Datasets

| Dataset | Scenes | Queries | Primary Variant Type | Variants |
|---------|--------|---------|---------------------|----------|
| R1 | 10 | 500 | Structural compression | 7 |
| R2 | 10 | 500 | Dimension ablation + structural | 11 |
| conflict | 5 | 250 | Base + conflict | 7 |

## Reproducibility

```bash
# Generate complete dataset (deterministic)
python tools/generate_dataset.py --output_dir ./data_set --seed 42

# Validate
python tools/validate_dataset.py --base_dir ./data_set --validate --check_equivalence

# Run tests
python tests/test_scene_serializer.py
python tests/test_generate_dataset.py

# Aggregate evaluation results
python tools/aggregate_results.py
```

## Compliance with Rebuttal Requirements

| Claim | Status | Implementation |
|-------|--------|----------------|
| "Rule-based serialization automatically generates all data formats" | FULFILLED | `tools/scene_serializer.py` — pure Python with zero API calls |
| "Flat, hierarchical, clustered with 0%/50%/75% compression" | FULFILLED | All 7 variants in `generate_all_variants()` |
| "Dimension-ablation variants" | FULFILLED | 4 variants in `generate_dimension_variants()` |
| "Conflict variants" | FULFILLED | 3 conflict variants in `generate_conflict_variants()` |
| "Information equivalence guarantee" | FULFILLED | Verified for all 25 scenes |
| "No human/LLM annotation" | FULFILLED | 100% rule-based generation |
