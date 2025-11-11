# Converter Tools

[English](#) | [中文](converter_cn.md)

Trace retrieval, conversion, and quality checking tools for NexGAP pipeline.

---

## Overview

The converter module provides three main capabilities:

1. **Trace Retrieval** - Fetch execution traces from Langfuse
2. **Format Conversion** - Convert traces to training formats (OpenAI, NexAU XML, etc.)
3. **Quality Checking** - Validate and filter converted data

---

## Architecture

```
converter/
├── trace/                          # Trace operations
│   ├── get_trace.py               # Single trace retrieval
│   ├── get_traces.py              # Batch retrieval
│   ├── convert_spans_to_chatcompletion.py      # OpenAI format
│   ├── convert_spans_to_chatcompletion_nexau.py # NexAU XML format
│   ├── convert_trace_tool_calls.py            # Tool call conversion
│   └── filter_xml_errors.py       # XML validation
├── mcp_preloader.py               # MCP tool cache
└── cache_manager.py               # Cache management
```

---

## Usage

### 1. Trace Retrieval

**Single Trace:**
```bash
cd converter/trace
uv run get_trace.py <trace_id> -o output.jsonl
```

**Batch Retrieval:**
```bash
uv run get_traces.py \
  --start "2025-11-01 00:00:00" \
  --end "2025-11-14 23:59:59" \
  --output-dir ./traces \
  --concurrency 4
```

### 2. Format Conversion

**Convert to OpenAI Format:**
```bash
uv run convert_spans_to_chatcompletion.py \
  trace.jsonl \
  -o output.jsonl \
  -c ../../Gyrfalcon/frameworks/my_framework/framework_config.yaml
```

**Convert to NexAU XML Format:**
```bash
uv run convert_spans_to_chatcompletion_nexau.py \
  trace.jsonl \
  -o output.jsonl
```

**Convert Tool Call Format:**
```bash
uv run convert_trace_tool_calls.py \
  input.jsonl \
  -o output.jsonl \
  --format qwen  # Options: qwen, minimax, glm, deepseek
```

### 3. Quality Checking

**XML Validation:**
```bash
uv run filter_xml_errors.py input.jsonl output.jsonl
```

Validates:
- `<tool_use>` blocks
- `<parallel_tool>` structures
- `<sub-agent>` blocks
- Tag balance and nesting

---

## Configuration

### Langfuse Setup

Required in `.env`:
```bash
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Conversion Options

**Environment variables:**
```bash
USE_OPENAI_CONVERTER=true    # Use OpenAI format (default: NexAU XML)
USE_ANTHROPIC_API=true       # For Anthropic spans
```

**Command-line flags** (recommended for `run_end_to_end.py`):
```bash
--use-openai-format          # Convert to OpenAI format
--tool-call-format qwen      # Specific model format
```

---

## Output Formats

### NexAU XML Format (Default)

```json
{
  "messages": [
    {"role": "user", "content": "Query"},
    {"role": "assistant", "content": "<tool_use>\n<tool_name>search</tool_name>\n<parameter>...</parameter>\n</tool_use>"},
    {"role": "tool", "content": "Result"},
    {"role": "assistant", "content": "Response"}
  ],
  "tools": [...],
  "trace_id": "...",
  "framework": "..."
}
```

### OpenAI Format

```json
{
  "messages": [
    {"role": "user", "content": "Query"},
    {"role": "assistant", "tool_calls": [{"id": "...", "function": {...}}]},
    {"role": "tool", "tool_call_id": "...", "content": "Result"},
    {"role": "assistant", "content": "Response"}
  ],
  "tools": [...]
}
```

---

## Cache Management

The converter uses caching to improve performance:

**MCP Tool Cache:**
- Location: `converter/.cache/mcp_tools_cache.json`
- Stores MCP tool definitions
- Auto-updated on first load

**Cache Operations:**
```python
from converter import get_cache_manager

cache_mgr = get_cache_manager()

# View stats
stats = cache_mgr.get_cache_stats()

# Clear cache (after config updates)
cache_mgr.clear_mcp_cache()
```

---

## Performance

**Cache Performance:**
- First run: ~30 seconds (fetching MCP tools)
- Cached runs: ~0.1 seconds
- Cache size: ~100KB

**Batch Processing:**
- Use `--concurrency 4` for parallel trace retrieval
- Use `--max-workers 5-10` in `run_end_to_end.py`

---

## Common Workflows

### Full Pipeline (Recommended)

Use `run_end_to_end.py` which handles everything:

```bash
uv run run_end_to_end.py \
  --query-filepath queries.jsonl \
  --output-dir output/ \
  --max-workers 5
```

Output: `output/framework/converted_trace/*.jsonl`

### Manual Pipeline

```bash
# 1. Fetch traces
cd converter/trace
uv run get_traces.py \
  --start "2025-11-01" \
  --end "2025-11-14" \
  --output-dir traces/

# 2. Convert
for file in traces/*/*.jsonl; do
  uv run convert_spans_to_chatcompletion.py "$file" -o "${file%.jsonl}_chat.jsonl"
done

# 3. Validate
uv run filter_xml_errors.py input.jsonl output.jsonl

# 4. Merge
cat traces/*/*_chat.jsonl > training_data.jsonl
```

---

## Troubleshooting

**Empty Output Files:**
- Check `USE_ANTHROPIC_API` setting
- Verify framework config path with `-c`
- Use `--verbose` for debug info

**Missing Tools:**
- Provide framework config: `-c framework_config.yaml`
- Clear cache: `cache_mgr.clear_mcp_cache()`

**Langfuse Connection:**
```bash
# Verify credentials
cat .env | grep LANGFUSE
```

---

## API Reference

### MCPPreloader

```python
from converter import MCPPreloader, get_cache_manager

cache_mgr = get_cache_manager()
preloader = MCPPreloader(base_dir, cache_mgr)
cache = preloader.preload_all_configs(use_cache=True)
```

### XMLValidator

```python
from converter.trace.filter_xml_errors import XMLValidator

validator = XMLValidator(mode='a4a')  # or 'nexau'
is_valid, errors = validator.validate_message(content)
```

---

## See Also

- [Workflow Guide](workflow.md) - Full pipeline overview
- [Gyrfalcon Usage](gyrfalcon.md) - Query generation
- [Main README](../README.md) - Installation and setup
