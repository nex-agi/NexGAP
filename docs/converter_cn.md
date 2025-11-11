# Converter Tools

[English](converter.md) | [中文](#)

NexGAP pipeline 中的 trace 检索、转换和质量检查工具。

---

## 概览

Converter 模块提供三大核心能力：

1. **Trace 检索** - 从 Langfuse 获取执行 traces
2. **格式转换** - 转换为训练格式（OpenAI、NexAU XML 等）
3. **质量检查** - 验证和过滤转换数据

---

## 架构

```
converter/
├── trace/                          # Trace 操作
│   ├── get_trace.py               # 单个 trace 检索
│   ├── get_traces.py              # 批量检索
│   ├── convert_spans_to_chatcompletion.py      # OpenAI 格式
│   ├── convert_spans_to_chatcompletion_nexau.py # NexAU XML 格式
│   ├── convert_trace_tool_calls.py            # Tool call 转换
│   └── filter_xml_errors.py       # XML 验证
├── mcp_preloader.py               # MCP tool cache
└── cache_manager.py               # Cache 管理
```

---

## 使用

### 1. Trace 检索

**单个 Trace：**
```bash
cd converter/trace
uv run get_trace.py <trace_id> -o output.jsonl
```

**批量检索：**
```bash
uv run get_traces.py \
  --start "2025-11-01 00:00:00" \
  --end "2025-11-14 23:59:59" \
  --output-dir ./traces \
  --concurrency 4
```

### 2. 格式转换

**转换为 OpenAI 格式：**
```bash
uv run convert_spans_to_chatcompletion.py \
  trace.jsonl \
  -o output.jsonl \
  -c ../../Gyrfalcon/frameworks/my_framework/framework_config.yaml
```

**转换为 NexAU XML 格式：**
```bash
uv run convert_spans_to_chatcompletion_nexau.py \
  trace.jsonl \
  -o output.jsonl
```

**转换 Tool Call 格式：**
```bash
uv run convert_trace_tool_calls.py \
  input.jsonl \
  -o output.jsonl \
  --format qwen  # 选项：qwen, minimax, glm, deepseek
```

### 3. 质量检查

**XML 验证：**
```bash
uv run filter_xml_errors.py input.jsonl output.jsonl
```

验证内容：
- `<tool_use>` 块
- `<parallel_tool>` 结构
- `<sub-agent>` 块
- 标签平衡和嵌套

---

## 配置

### Langfuse 设置

`.env` 中需要配置：
```bash
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 转换选项

**环境变量：**
```bash
USE_OPENAI_CONVERTER=true    # 使用 OpenAI 格式（默认：NexAU XML）
USE_ANTHROPIC_API=true       # 用于 Anthropic spans
```

**命令行参数**（在 `run_end_to_end.py` 中使用）：
```bash
--use-openai-format          # 转换为 OpenAI 格式
--tool-call-format qwen      # 特定模型格式
```

---

## 输出格式

### NexAU XML 格式（默认）

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

### OpenAI 格式

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

## Cache 管理

Converter 使用缓存来提高性能：

**MCP Tool Cache：**
- 位置：`converter/.cache/mcp_tools_cache.json`
- 存储 MCP tool 定义
- 首次加载时自动更新

**Cache 操作：**
```python
from converter import get_cache_manager

cache_mgr = get_cache_manager()

# 查看统计
stats = cache_mgr.get_cache_stats()

# 清除缓存（配置更新后）
cache_mgr.clear_mcp_cache()
```

---

## 性能

**Cache 性能：**
- 首次运行：~30 秒（获取 MCP tools）
- 使用缓存：~0.1 秒
- Cache 大小：~100KB

**批量处理：**
- 使用 `--concurrency 4` 进行并行 trace 检索
- 在 `run_end_to_end.py` 中使用 `--max-workers 5-10`

---

## 常用工作流程

### 完整 Pipeline（推荐）

使用 `run_end_to_end.py` 处理所有步骤：

```bash
uv run run_end_to_end.py \
  --query-filepath queries.jsonl \
  --output-dir output/ \
  --max-workers 5
```

输出：`output/framework/converted_trace/*.jsonl`

### 手动 Pipeline

```bash
# 1. 获取 traces
cd converter/trace
uv run get_traces.py \
  --start "2025-11-01" \
  --end "2025-11-14" \
  --output-dir traces/

# 2. 转换
for file in traces/*/*.jsonl; do
  uv run convert_spans_to_chatcompletion.py "$file" -o "${file%.jsonl}_chat.jsonl"
done

# 3. 验证
uv run filter_xml_errors.py input.jsonl output.jsonl

# 4. 合并
cat traces/*/*_chat.jsonl > training_data.jsonl
```

---

## 故障排除

**输出文件为空：**
- 检查 `USE_ANTHROPIC_API` 设置
- 使用 `-c` 验证 framework config 路径
- 使用 `--verbose` 查看调试信息

**缺少 Tools：**
- 提供 framework config：`-c framework_config.yaml`
- 清除缓存：`cache_mgr.clear_mcp_cache()`

**Langfuse 连接问题：**
```bash
# 验证凭证
cat .env | grep LANGFUSE
```

---

## API 参考

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

validator = XMLValidator(mode='a4a')  # 或 'nexau'
is_valid, errors = validator.validate_message(content)
```

---

## 相关文档

- [工作流程指南](workflow_cn.md) - 完整 pipeline 概览
- [Gyrfalcon 使用](gyrfalcon_cn.md) - Query 生成
- [主文档](README_cn.md) - 安装和设置
