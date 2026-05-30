# Dataset Documentation

Detailed information about the dataset structure, format variants, and query types.

## Overview

| Dataset | Scenes | Rooms | Queries/Scene | Total Queries | Variants/Scene |
|:---|:---:|:---:|:---:|:---:|:---:|
| **R1** (Simple) | 10 | 4-6 | 50 | 500 | 18 |
| **R2** (Complex) | 10 | 6-14 | 50 | 500 | 22 |
| **Conflict** | 10 | 6-14 | 50 | 500 | 7 |
| **Total** | **30** | | | **1,500** | |

**Generation method**: Fully automated, rule-based, deterministic. No human annotation or LLM-based annotation.

## Scene JSON Schema

```json
{
  "scene_id": "scene_complex_01",
  "description": "Complex scene 01: G1 - Basic cognition (6 rooms)",
  "world": {
    "scene_name": "scene_complex_01",
    "gradient": "G1",
    "gradient_label": "Basic cognition",
    "rooms": [
      {
        "idx": 0,
        "room_id": "R1",
        "canonical": "Library",
        "synonyms": ["Reading Room", "Archive"],
        "abbr": "Lib",
        "attributes": ["Bright"],
        "x": 0, "y": 0, "w": 4, "h": 4
      }
    ],
    "edges": [[0, 1], [1, 2]],
    "objects": {
      "0": [
        {
          "obj_id": "O1",
          "canonical": "Printer",
          "synonyms": ["Scanner", "Copier"],
          "abbr": "Prn",
          "attributes": ["Large"],
          "name": "Printer_1",
          "display_name": "Printer"
        }
      ]
    },
    "history": [
      {"object": "Key", "from_room": "R1", "to_room": "R2"}
    ],
    "containment": {"Printer_1": "R1"},
    "parallel_rooms": [["R1", "R2"]],
    "parallel_objects": [],
    "rules": ["When generating a path, output the shortest valid path..."]
  },
  "variants": {
    "flat": "Start at Bright Library R1(0,0). Large Printer...",
    "flat_50": "...",
    "flat_25": "Lib:R1(0,0)[Prn] | ...",
    "hier": "[R1: Library [Bright] @ (0,0), Printer(Printer)]",
    "hier_50": "...",
    "hier_25": "...",
    "clustered": "Zone A: {R1,R2} @ (2,0) ...",
    "clustered_50": "...",
    "clustered_25": "..."
  }
}
```

### Room Fields

| Field | Type | Description |
|:---|:---|:---|
| `idx` | int | Zero-based index |
| `room_id` | str | Human-readable ID (R1, R2, ...) |
| `canonical` | str | Canonical name from knowledge base |
| `synonyms` | list[str] | At least 2 synonyms |
| `abbr` | str | Abbreviation (e.g., "Lib") |
| `attributes` | list[str] | Room attributes (e.g., "Bright", "Quiet") |
| `x`, `y` | int | Grid coordinates |
| `w`, `h` | int | Room dimensions |

### Object Fields

| Field | Type | Description |
|:---|:---|:---|
| `obj_id` | str | Object type ID (O1, O2, ...) |
| `canonical` | str | Canonical name from knowledge base |
| `synonyms` | list[str] | At least 2 synonyms |
| `abbr` | str | Abbreviation |
| `attributes` | list[str] | Object attributes (e.g., "Red", "Large") |
| `name` | str | Instance name (e.g., "Printer_1") |
| `display_name` | str | Display name (may differ from canonical) |

## Format Variants

### Structural Variants (R1, R2)

Each scene has 9 structural variants: 3 formats x 3 retention levels.

#### Flat Format

```
100%: Start at Bright Library R1(0,0). Large Printer, Blue Scanner. Go east from R1 to Quiet Kitchen R2(4,0). Small Table. Before, the Key moved from R1 to R2.
 50%: Start at Bright Library R1(0,0). Large Printer, Blue Scanner. Go east from R1 to Quiet Kitchen R2(4,0). Small Table. Edges: R1-R2. Key moved: R1->R2
 25%: Lib:R1(0,0)[Prn,Mon] | Kit:R2(4,0)[Tbl] | Edg:1-2 | Key:R1->R2
```

#### Hierarchical Format

```
100%: [R1: Library [Bright] @ (0,0), Printer(Printer, Scanner)]
      [R2: Kitchen [Quiet] @ (4,0), Table(Table)]
      Topo: R1-R2
      Hist: Key moved: R1->R2
 50%:  [HIERARCHY]
       Scene(01) -> R1,R2
       R1(0,0): Bright Library {Prn,Mon}
       R2(4,0): Quiet Kitchen {Tbl}
       Edges: R1-R2
       Key: R1->R2
 25%:  C01[R1(0,0):Lib[Prn,Mon]; R2(4,0):Kit[Tbl]] | Edg:1-2 | Key:R1->R2
```

#### Clustered Format

```
100%: Zone A: {R1,R2} @ (2,0) Link: A<->B Key moved: R1->R2
 50%:  [CLUSTERS]
       A(R1,R2) @(2,0): Bright {Prn/Mon/Tbl}
       B(R3,R4) @(6,4): Quiet {Cab}
       [LINKS] Zone_A <-> Zone_B via R2-R3
       [TRACE] Key: R1->R4
 25%:  G1(2,0):1,2. G2(6,4):3,4. Links:G1-G2(2-3). Key:R1->R4.
```

### Dimension-Ablation Variants (R2)

4 variants that selectively remove spatial dimensions:

| Variant | Topology | Geometry | Semantics | History | Rules |
|:---|:---:|:---:|:---:|:---:|:---:|
| `flat_full` | Y | Y | Y | Y | Y |
| `flat_topo_hist` | Y | - | - | Y | - |
| `flat_geom_rule_hist` | - | Y | - | Y | Y |
| `flat_sem_rule_hist` | - | - | Y | Y | Y |

### Semantic Variation Variants (R1, R2)

9 variants where canonical names are replaced with synonyms from the knowledge base. Topology, geometry, and history remain unchanged.

Example: "Library" -> "Reading Room", "Printer" -> "Scanner"

### Conflict Variants (Conflict)

| Variant | Description |
|:---|:---|
| `conflict_flat` | Flat format with duplicate labels for distinct rooms |
| `conflict_hier` | Hierarchical format with duplicate labels |
| `conflict_clustered` | Clustered format with duplicate labels |

Semantic conflict (C2): Multiple distinct rooms share the same canonical label. IDs, topology, geometry, and history remain correct.

## Query Types

### ObjectLocation
```
Q: Where is the Printer currently located?
A: R3
```

### GeometryYN
```
Q: Is the Library (R1) located North the Kitchen (R2)?
A: YES
```

### TopologyYN
```
Q: Is there a direct connection between R1 and R3?
A: NO
```

### ReachabilityYN
```
Q: Can you reach R4 from R1?
A: YES
```

### PathGen
```
Q: Provide the shortest path from R1 to R4.
A: R1->R2->R3->R4
```

## Knowledge Base

### Room Categories (12)

Each room has a canonical name, at least 2 synonyms, and an abbreviation.

| Canonical | Synonyms | Abbr |
|:---|:---|:---|
| Office | Workspace, Bureau | Off |
| Corridor | Aisle, Hallway | Corr |
| Meeting Room | Conference Room, Boardroom | MR |
| Lobby | Entrance Hall, Foyer | Lob |
| Bedroom | Sleeping Quarters, Chamber | Bed |
| Kitchen | Pantry, Cookhouse | Kit |
| Laboratory | Lab, Research Room | Lab |
| Storage Room | Storeroom, Stockroom | Stor |
| Bathroom | Restroom, Washroom | Bath |
| Classroom | Lecture Room, Tutorial Room | CR |
| Entrance | Entryway, Doorway | Ent |
| Library | Reading Room, Archive | Lib |

### Object Categories (10)

| Canonical | Synonyms | Abbr |
|:---|:---|:---|
| Chair | Seat, Stool | Chr |
| Desk | Table, Workbench | Dsk |
| Laptop | Notebook Computer, PC | Lap |
| Table | Counter, Bench | Tbl |
| Sofa | Couch, Settee | Sof |
| Key | Passkey, Access Card | Key |
| Cabinet | Cupboard, Locker | Cab |
| Printer | Scanner, Copier | Prn |
| Monitor | Display, Screen | Mon |
| Bookshelf | Shelf, Rack | Bks |

### Attributes

**Room attributes**: Bright, Dark, Messy, Clean, Quiet, Crowded, Spacious, Narrow

**Object attributes**: Red, Blue, Large, Small, Heavy, Light, Wooden, Metallic

## Information Equivalence

All format variants encode the same underlying spatial truths:

1. **Room identity**: All room IDs (or numeric indices) appear in every variant
2. **Topology**: Connection information is encoded (explicitly or implicitly)
3. **History**: Object movement trace is preserved
4. **Semantics**: Room types and objects are represented

Verify with:
```bash
python tools/validate_dataset.py --base_dir ./data_set --check_equivalence
```

## Reproducibility

```bash
# Same seed -> identical dataset
python tools/generate_dataset.py --seed 42

# Verify determinism
diff <(python tools/generate_dataset.py --seed 42 --output_dir /tmp/v1 2>&1) \
     <(python tools/generate_dataset.py --seed 42 --output_dir /tmp/v2 2>&1)
```
