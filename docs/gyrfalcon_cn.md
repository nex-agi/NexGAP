# Gyrfalcon Query Synthesis

[English](gyrfalcon.md) | [中文](#)

基于问题类型树为 AI agent frameworks 生成高质量测试 queries。

---

## 概览

Gyrfalcon 是一个智能 query 生成管道，基于问题类型树创建多样化的测试查询，支持难度控制、多语言生成和并行执行。

**核心特性：**
- 基于问题类型树生成
- 难度控制（easy、medium、hard）
- 多语言支持（English、Chinese）
- 支持并行执行
- Web 搜索增强（可选）

---

## 快速开始

```bash
cd Gyrfalcon

# 为 framework 生成 queries
uv run main.py \
  --framework my_framework \
  --num-queries 50 \
  --language english

# 输出：output/my_framework_queries_YYYYMMDD_HHMMSS.jsonl
```

---

## 使用

### 基本命令

```bash
uv run main.py \
  --framework <framework_name> \
  --num-queries <number> \
  --language <english|chinese>
```

### 高级选项

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

**参数说明：**

| 参数 | 说明 | 默认值 |
|-----------|-------------|---------|
| `--framework` | Framework 名称（必须存在于 `frameworks/` 中） | 必需 |
| `--num-queries` | 要生成的 query 数量 | `10` |
| `--language` | Query 语言：`english` 或 `chinese` | `english` |
| `--difficulty-distribution` | Easy、Medium、Hard 比例（格式：easy:0.2,medium:0.5,hard:0.3） | `easy:0.2,medium:0.5,hard:0.3` |
| `--num-workers` | 并行生成的 worker 数 | `1` |
| `--websearch-prob` | Web 搜索增强概率（0.0-1.0） | `0.0` |
| `--output-dir` | 自定义输出目录 | `output/` |

---

## 配置

### 1. Framework Config 设置

Gyrfalcon 需要以下 framework 配置：

```bash
frameworks/
└── my_framework/
    ├── config.json                 # Framework 核心配置
    ├── framework_config.yaml       # 元数据和描述
    ├── problem_types.json          # 问题类型树
    └── persona.jsonl               # 用户 personas（500 条）
```

**自动生成配置：**使用 GyrfalconFrameworkGenerator
```bash
cd NexA4A
uv run agent4agent.py interactive
# → 使用 GyrfalconFrameworkGenerator sub-agent
```

### 2. 环境变量

```bash
# LLM 配置（必需）
LLM_API_KEY=your-api-key
LLM_BASE_URL=your-base-url
LLM_MODEL=your-model

# 可选设置
LLM_TEMPERATURE=0.6
LLM_MAX_TOKENS=16000
LLM_TIMEOUT=600.0
LLM_MAX_RETRIES=10

# Web Search（可选）
SERPER_API_KEY=your-serper-key
```

---

## 问题类型树

问题类型树定义了 framework 的查询分类：

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

**树结构：**
- **根节点**：Framework 领域
- **分支节点**：类别
- **叶子节点**：具体问题类型

**采样策略：**
- 叶子节点：直接生成 query
- 分支节点：从子节点递归采样
- 确保问题类型覆盖的多样性

---

## 输出格式

生成的 queries 保存为 JSONL 格式：

```json
{
  "query": "患者出现持续发热和咳嗽，如何诊断？",
  "difficulty": "medium",
  "problem_type": "Medical Diagnosis > Symptom Analysis > Acute Symptoms",
  "framework": "medical_assistant",
  "language": "chinese"
}
```

---

## 难度等级

**Easy（简单）：**
- 单步问题
- 明确的需求
- 直接的解决方案

**Medium（中等）：**
- 多步推理
- 一定的模糊性
- 需要使用 tools

**Hard（困难）：**
- 复杂场景
- 多个约束条件
- 边界情况和异常

**控制分布：**
```bash
--difficulty-distribution easy:0.2,medium:0.5,hard:0.3  # 20% easy, 50% medium, 30% hard
```

---

## 性能

**生成速度：**
- 单 worker：~10 queries/分钟
- 5 workers：~40 queries/分钟
- 线性扩展

**推荐设置：**
- 开发阶段：`--num-workers 1`，`--num-queries 20`
- 生产阶段：`--num-workers 4-8`，`--num-queries 100+`

---

## Web Search 增强

启用 web search 来丰富 query 上下文：

```bash
# 1. 在 .env 中设置 API key
SERPER_API_KEY=your-key

# 2. 在命令中启用（设置概率 > 0）
uv run main.py \
  --framework my_framework \
  --num-queries 50 \
  --websearch-prob 0.5  # 50% 的 queries 将使用 web search
```

**优势：**
- 真实世界的上下文
- 时事热点
- 领域特定术语
- 增强 query 多样性

---

## 常用工作流程

### 开发工作流程

```bash
# 1. 在 Agent4Agent 中创建 framework
cd NexA4A
uv run agent4agent.py interactive

# 2. 生成 Gyrfalcon config
# 使用 GyrfalconFrameworkGenerator sub-agent

# 3. 测试 query 生成（小批量）
cd ../Gyrfalcon
uv run main.py \
  --framework my_framework \
  --num-queries 10 \
  --language english

# 4. 查看输出
cat output/my_framework_queries_*.jsonl | jq .
```

### 生产工作流程

```bash
# 生成大量 query 集
uv run main.py \
  --framework my_framework \
  --num-queries 500 \
  --language english \
  --difficulty-distribution easy:0.2,medium:0.5,hard:0.3 \
  --num-workers 8 \
  --websearch-prob 0.3 \
  --output-dir production_queries/

# 在 pipeline 中使用
cd ..
uv run run_end_to_end.py \
  --query-filepath Gyrfalcon/production_queries/*.jsonl \
  --output-dir output/experiment_01
```

---

## 目录结构

```
Gyrfalcon/
├── main.py                     # 入口点
├── frameworks/                 # Framework configs
│   └── <framework_name>/
│       ├── framework_config.yaml
│       ├── problem_type_tree.json
│       └── queries_example.jsonl
├── llm_interface/             # LLM 集成
│   └── query_generator.py
├── problem_type_tree/         # Tree 管理
│   ├── tree_manager.py
│   ├── sampling_manager.py
│   └── visualizer.py
└── output/                    # 生成的 queries
```

---

## 故障排除

**没有生成 queries：**
- 检查 framework config 是否存在：`frameworks/<name>/`
- 验证 `.env` 中的 LLM 凭证
- 检查问题类型树结构

**生成的 Query 质量不佳：**
- 调整难度分布
- 启用 web search 增强
- 查看 query 示例
- 优化问题类型树

**性能问题：**
- 增加 `--num-workers`
- 检查 LLM API 速率限制
- 减少每批次的 `--num-queries`

---

## 高级特性

### 自定义采样策略

在配置中修改采样行为：

```yaml
# framework_config.yaml
sampling:
  strategy: weighted  # 或：uniform, depth_first
  leaf_preference: 0.7
  max_depth: 3
```

### Persona Templates

提供多样化的用户 personas（自动生成 500 条）：

```jsonl
{"persona_en": "A 45-year-old family doctor in rural area", "persona_zh": "一位45岁的乡村家庭医生"}
{"persona_en": "A medical student preparing for exams", "persona_zh": "一位准备考试的医学生"}
```

位于 `frameworks/<name>/persona.jsonl`

---

## API 参考

### 主接口

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

## 相关文档

- [工作流程指南](workflow_cn.md) - 完整 pipeline 概览
- [Converter 工具](converter_cn.md) - Trace 处理
- [主文档](README_cn.md) - 安装和设置
