# LHAW Synthetic Underspecification Pipeline

Generate underspecified task variants from well-specified prompts.

**Domains:** TAC (enterprise) | SWE-Bench Pro (code repair) | MCP-Atlas (tool-use)

---

## Quick Start

```python
from synthetic import SyntheticPipeline

pipeline = SyntheticPipeline()

# TAC tasks (uses trajectory grounding for criticality)
result = pipeline.process_tac_task("finance_check_attendance_payroll")

# SWE-Bench Pro tasks
result = pipeline.process_swebench_task("django__django-12345")

# MCP Atlas tasks (auto-loads trajectory from CSV)
result = pipeline.process_mcpatlas_task("689f4d693e212e8ef3390720")

# Access variants
for variant in result.variants:
    print(variant.underspecified_prompt)
    
    # What was removed (the latent dimension)
    seg = variant.removed_segments[0]
    print(f"  Removed: {seg.dimension.value} = {seg.value}")
    print(f"  Grounded: {seg.is_used_in_trajectory}")  # Only if trajectory provided
    print(f"  Affects: {seg.checkpoint_refs}")          # Only if checkpoints provided

# Export as JSON
output = result.to_dict()
```

### Pipeline Output Format

Each variant has **removed_segments** with optional grounding:

```jsonc
{
  "removed_segments": [
    {
      "dimension": "input",
      "value": "april-attendance-data.csv",
      // 3-level scale: 0.0 | 0.5 | 1.0
      "criticality": 1.0,   // 1.0 = task FAILS without this
      "guessability": 0.0,  // 0.0 = cannot recover (arbitrary filename)
      
      // Only present if trajectory_path provided
      "is_used_in_trajectory": true,
      "first_use_pct": 0.15,
      
      // Only present if checkpoints provided
      "checkpoint_refs": ["CP1", "CP2"]
    }
  ]
}
```

### Final Benchmark Format (after pass@k validation)

Analysis fields are added **after pass@k baselines** (not by the pipeline):

```jsonc
{
  "task_id": "...",
  "agent_prompt": "...",
  "removed_segments": [...],
  
  // From pipeline
  "expected_questions": ["Which file?"],
  "expected_failure_mode": "...",
  
  // Added AFTER pass@k analysis
  "ambiguity_class": "outcome-critical",
  "oracle_decision": "clarify"
}
```

### Ablation Levers

```python
from synthetic import SyntheticPipeline, Severity

pipeline = SyntheticPipeline()

# TAC tasks
result = pipeline.process_tac_task("finance_check_attendance_payroll")

# Access variants
for variant in result.variants:
    print(variant.underspecified_prompt)
    seg = variant.removed_segments[0]
    print(f"  Removed: [{seg.dimension.value}] {seg.value}")
```

---

## Core Data Structures

```
synthetic/
├── __init__.py          # Package exports
├── core.py              # Canonical types: Segment, Blocker, UnderspecVariant, etc.
├── pipeline.py          # Unified SyntheticPipeline
├── llm.py               # LLM completion utilities
├── adapters/
│   ├── base.py          # BaseTaskAdapter (abstract)
│   ├── tac.py           # TAC → UnifiedTask
│   ├── swebench.py      # SWE-Bench Pro → UnifiedTask
│   └── mcpatlas.py      # MCP Atlas → UnifiedTask (auto-loads trajectory from CSV)
├── configs/
│   └── taxonomy.yaml    # Full underspecification taxonomy (loaded into LLM prompts)
└── notebooks/
    ├── underspec_comparison.ipynb  # Underspec variant comparison
    └── ask_user_exploration.ipynb  # Ask-user behavior exploration
```

### Segment

Atomic piece of information that can be removed to create underspecification.

```python
@dataclass
class Segment:
    id: str                          # "S1", "S2", ...
    dimension: Dimension             # GOAL | CONSTRAINT | INPUT | CONTEXT
    subdimension: str                # From taxonomy (e.g., "identifier", "format")
    value: Any                       # Extracted specific value
    text: str                        # Full text span in prompt
    
    # Grounding (from trajectory/checkpoints)
    is_used_in_trajectory: bool      # Was value used in golden trajectory?
    first_use_pct: float             # When first used (0.0=start, 1.0=end)
    checkpoint_refs: List[str]       # Which checkpoints affected
    
    # Scores
    criticality: float               # 0.0 (OK) | 0.5 (WRONG) | 1.0 (FAILS)
    guessability: float              # 0.0 (cannot) | 0.5 (maybe) | 1.0 (will)
    priority_score: float            # criticality × (1 - guessability)
    
    # Dataset-specific metadata
    metadata: Dict[str, Any]         # e.g., {"source_field": "task_instructions"}
```

### UnderspecVariant

A modified task with controlled underspecification.

```python
@dataclass
class UnderspecVariant:
    id: str
    original_prompt: str
    underspecified_prompt: str
    removed_segments: List[Segment]
    severity: Severity               # DELETE | VAGUIFY | GENERICIZE
    
    # For evaluation
    expected_failure_mode: str
    expected_questions: List[Dict]   # [{segment_id, questions}]
    predicted_difficulty: float      # mean priority of removed segments
```

### Enums

```python
class Dimension(Enum):
    GOAL = "goal"           # WHAT to produce
    CONSTRAINT = "constraint"  # HOW to do it
    INPUT = "input"         # FROM WHERE
    CONTEXT = "context"     # WHAT BACKGROUND

class Severity(Enum):
    DELETE = "delete"       # Remove entirely (HIGH)
    VAGUIFY = "vaguify"     # Vague language (MEDIUM)
    GENERICIZE = "genericize"  # Subtle rewording (LOW)
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SEGMENT EXTRACTION                                         │
├─────────────────────────────────────────────────────────────────────┤
│ Input: prompt + taxonomy + trajectory (opt) + checkpoints (opt)     │
│ Output: segments with dimension, criticality, guessability          │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: VARIANT GENERATION                                         │
├─────────────────────────────────────────────────────────────────────┤
│ Severity: DELETE | VAGUIFY | GENERICIZE                             │
│ Output: underspecified_prompt + expected_questions + failure_mode   │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: EMPIRICAL VALIDATION (pass@k)                              │
├─────────────────────────────────────────────────────────────────────┤
│ 0/N + divergent states  → OUTCOME-CRITICAL (keep)                   │
│ Some success + variable → DIVERGENT (keep)                          │
│ N/N consistent success  → BENIGN (keep)                             │
│ 0/N + 1 state (LLM: ✓) → NEW_TASK (filter out)                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Segments are latent dimensions. Empirical trials ground ambiguity in execution outcomes rather than linguistic intuition.

---

## Output Format

### Phase 2 (Post-Generation)

```jsonc
{
  "task_id": "finance_check_attendance_payroll_V_S3",
  "agent_prompt": "Create payroll using the attendance file...",
  "removed_segments": [{
    "id": "S3",
    "dimension": "input",
    "subdimension": "identifier",
    "value": "april-attendance-data.csv",
    "criticality": 1.0,
    "guessability": 0.5,
    "priority_score": 0.5,
    "is_used_in_trajectory": true,
    "checkpoint_refs": ["CP1", "CP2"]
  }],
  "criteria": { "severity": "delete" },
  "expected_questions": [{"S3": ["Which attendance file should I use?"]}],
  "expected_failure_mode": "Agent uses wrong file, produces incorrect payroll"
}
```

### Phase 3 (Benchmark-Ready)

```jsonc
[
  {
    "variant_id": "hr_new_grad_job_description_V10_goal",
    "original_task": "hr_new_grad_job_description",
    "dataset": "TheAgentCompany",
    "original_prompt": "Write a new grad software engineer job...",
    "underspecified_prompt": "Write a job description...",
    "information_dimension": "goal",
    "ambiguity_class": "benign",
    "removed_segments": [
      {"id": "S1", "dimension": "goal", "subdimension": "target",
       "value": "new grad software engineer job description"}
    ],
    "expected_questions": [
      {"segment_id": "S1", "questions": ["What type of job description should I create?"]}
    ],
    "terminal_states": "[(1, 1)]"
  }
]
```

---

## File Structure

```
synthetic/
├── core.py              # Segment, Blocker, UnderspecVariant, UnifiedTask
├── pipeline.py          # SyntheticPipeline
├── llm.py               # LLM utilities
├── adapters/
│   ├── base.py          # BaseTaskAdapter
│   ├── tac.py           # TAC → UnifiedTask
│   ├── swebench.py      # SWE-Bench Pro → UnifiedTask
│   └── mcpatlas.py      # MCP Atlas → UnifiedTask
└── configs/
    └── taxonomy.yaml    # GOAL/CONSTRAINT/INPUT/CONTEXT definitions
```

---

## Ambiguity Classification

Based on pass@N empirical trials:

| Class | Definition | Oracle | Target % |
|-------|------------|--------|----------|
| **outcome-critical** | 0/N success + divergent terminal states | CLARIFY | 40% |
| **divergent** | Some success + variable outcomes | PROCEED | 30% |
| **benign** | N/N success despite missing info | PROCEED | 30% |
| **new_task** | 0/N + 1 state (LLM judged different task) | N/A | filtered |

> `new_task` variants are filtered out — they indicate the deletion changed the task goal rather than creating genuine ambiguity.
