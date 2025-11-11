# Gyrfalcon Query Synthesis

[English](#) | [中文](gyrfalcon_cn.md)

Generate high-quality, diverse test queries for AI agent frameworks using problem type trees.

---

## Overview

Gyrfalcon is an intelligent query generation pipeline that creates diverse test queries based on problem type trees. It supports difficulty control, multilingual generation, and parallel execution.

**Key Features:**
- Problem type tree-based generation
- Difficulty control (easy, medium, hard)
- Multilingual support (English, Chinese)
- Parallel execution for speed
- Web search enrichment (optional)

---

## Quick Start

```bash
cd Gyrfalcon

# Generate queries for a framework
uv run main.py \
  --framework my_framework \
  --num-queries 50 \
  --language english

# Output: output/my_framework_queries_YYYYMMDD_HHMMSS.jsonl
```

---

## Usage

### Basic Command

```bash
uv run main.py \
  --framework <framework_name> \
  --num-queries <number> \
  --language <english|chinese>
```

### Advanced Options

```bash
uv run main.py \
  --framework medical_assistant \
  --num-queries 100 \
  --language english \
  --difficulty-distribution easy:0.3,medium:0.4,hard:0.3 \
  --num-workers 5 \
  --websearch-prob 0.5 \
  --output-dir custom_output/
```

**Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--framework` | Framework name (must exist in `frameworks/`) | Required |
| `--num-queries` | Number of queries to generate | `10` |
| `--language` | Query language: `english` or `chinese` | `english` |
| `--difficulty-distribution` | Easy,Medium,Hard ratio (format: easy:0.2,medium:0.5,hard:0.3) | `easy:0.2,medium:0.5,hard:0.3` |
| `--num-workers` | Parallel workers for generation | `1` |
| `--websearch-prob` | Probability of web search enrichment (0.0-1.0) | `0.0` |
| `--output-dir` | Custom output directory | `output/` |

---

## Configuration

### 1. Framework Config Setup

Gyrfalcon requires framework configuration files:

```bash
frameworks/
└── my_framework/
    ├── config.json                 # Framework core configuration
    ├── framework_config.yaml       # Metadata and descriptions
    ├── problem_types.json          # Problem type tree
    └── persona.jsonl               # User personas (500 entries)
```

**Generate configs** using GyrfalconFrameworkGenerator:
```bash
cd NexA4A
uv run agent4agent.py interactive
# → Use GyrfalconFrameworkGenerator sub-agent
```

### 2. Environment Variables

```bash
# LLM Configuration (Required)
LLM_API_KEY=your-api-key
LLM_BASE_URL=your-base-url
LLM_MODEL=your-model

# Optional Settings
LLM_TEMPERATURE=0.6
LLM_MAX_TOKENS=16000
LLM_TIMEOUT=600.0
LLM_MAX_RETRIES=10

# Web Search (Optional)
SERPER_API_KEY=your-serper-key
```

---

## Problem Type Trees

Problem type trees define the taxonomy of queries for your framework:

```json
{
  "name": "Medical Diagnosis",
  "children": [
    {
      "name": "Symptom Analysis",
      "children": [
        {"name": "Acute Symptoms"},
        {"name": "Chronic Conditions"}
      ]
    },
    {
      "name": "Treatment Planning",
      "children": [
        {"name": "Medication"},
        {"name": "Therapy"}
      ]
    }
  ]
}
```

**Structure:**
- **Root node**: Framework domain
- **Branch nodes**: Categories
- **Leaf nodes**: Specific problem types

**Sampling strategy:**
- Leaf nodes: Direct query generation
- Branch nodes: Recursive sampling from children
- Ensures diverse coverage across problem types

---

## Output Format

Generated queries are saved as JSONL:

```json
{
  "query": "How to diagnose a patient with persistent fever and cough?",
  "difficulty": "medium",
  "problem_type": "Medical Diagnosis > Symptom Analysis > Acute Symptoms",
  "framework": "medical_assistant",
  "language": "english"
}
```

---

## Difficulty Levels

**Easy:**
- Single-step problems
- Clear requirements
- Straightforward solutions

**Medium:**
- Multi-step reasoning
- Some ambiguity
- Requires tool usage

**Hard:**
- Complex scenarios
- Multiple constraints
- Edge cases and exceptions

**Control distribution:**
```bash
--difficulty-distribution easy:0.2,medium:0.5,hard:0.3  # 20% easy, 50% medium, 30% hard
```

---

## Performance

**Generation Speed:**
- Single worker: ~10 queries/minute
- 5 workers: ~40 queries/minute
- Scales linearly with workers

**Recommended Settings:**
- Development: `--num-workers 1`, `--num-queries 20`
- Production: `--num-workers 4-8`, `--num-queries 100+`

---

## Web Search Enrichment

Enable web search to enrich query context:

```bash
# 1. Set API key in .env
SERPER_API_KEY=your-key

# 2. Enable in command (set probability > 0)
uv run main.py \
  --framework my_framework \
  --num-queries 50 \
  --websearch-prob 0.5  # 50% queries will use web search
```

**Benefits:**
- Real-world context
- Current events
- Domain-specific terminology
- Enhanced query diversity

---

## Common Workflows

### Development Workflow

```bash
# 1. Create framework in Agent4Agent
cd NexA4A
uv run agent4agent.py interactive

# 2. Generate Gyrfalcon config
# Use GyrfalconFrameworkGenerator sub-agent

# 3. Test query generation (small batch)
cd ../Gyrfalcon
uv run main.py \
  --framework my_framework \
  --num-queries 10 \
  --language english

# 4. Review output
cat output/my_framework_queries_*.jsonl | jq .
```

### Production Workflow

```bash
# Generate large query set
uv run main.py \
  --framework my_framework \
  --num-queries 500 \
  --language english \
  --difficulty-distribution easy:0.2,medium:0.5,hard:0.3 \
  --num-workers 8 \
  --websearch-prob 0.3 \
  --output-dir production_queries/

# Use in pipeline
cd ..
uv run run_end_to_end.py \
  --query-filepath Gyrfalcon/production_queries/*.jsonl \
  --output-dir output/experiment_01
```

---

## Directory Structure

```
Gyrfalcon/
├── main.py                     # Entry point
├── frameworks/                 # Framework configs
│   └── <framework_name>/
│       ├── framework_config.yaml
│       ├── problem_type_tree.json
│       └── queries_example.jsonl
├── llm_interface/             # LLM integrations
│   └── query_generator.py
├── problem_type_tree/         # Tree management
│   ├── tree_manager.py
│   ├── sampling_manager.py
│   └── visualizer.py
└── output/                    # Generated queries
```

---

## Troubleshooting

**No queries generated:**
- Check framework config exists: `frameworks/<name>/`
- Verify LLM credentials in `.env`
- Check problem type tree structure

**Low quality queries:**
- Adjust difficulty distribution
- Enable web search enrichment
- Review query examples
- Refine problem type tree

**Performance issues:**
- Increase `--num-workers`
- Check LLM API rate limits
- Reduce `--num-queries` per batch

---

## Advanced Features

### Custom Sampling Strategy

Modify sampling behavior in config:

```yaml
# framework_config.yaml
sampling:
  strategy: weighted  # or: uniform, depth_first
  leaf_preference: 0.7
  max_depth: 3
```

### Persona Templates

Provide diverse user personas (automatically generated 500 entries):

```jsonl
{"persona_en": "A 45-year-old family doctor in rural area", "persona_zh": "..."}
{"persona_en": "A medical student preparing for exams", "persona_zh": "..."}
```

Located in `frameworks/<name>/persona.jsonl`

---

## API Reference

### Main Interface

```python
from Gyrfalcon.main import run_pipeline

queries = run_pipeline(
    framework_name="my_framework",
    num_queries=50,
    language="english",
    difficulty_dist=[0.3, 0.4, 0.3],
    max_workers=5
)
```

### Tree Manager

```python
from Gyrfalcon.problem_type_tree import TreeManager

tree_mgr = TreeManager("frameworks/my_framework/problem_type_tree.json")
problem_types = tree_mgr.sample_leaf_nodes(n=10)
```

---

## See Also

- [Workflow Guide](workflow.md) - Full pipeline overview
- [Converter Tools](converter.md) - Trace processing
- [Main README](../README.md) - Installation and setup
