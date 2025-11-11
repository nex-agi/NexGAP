# NexGAP

[English](../README.md) | [中文](#)

**General Agentic Data Pipeline（通用 Agent 数据管道）**

一个端到端的管道，用于生成高质量的 agentic 训练数据。基于 NexAU agent framework 和 NexA4A 的 agent 构建能力，涵盖 agent 创建、问题合成、轨迹生成和 trace 处理。

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)

---

## ✨ 特性

- **Agent 创建**: 使用 NexA4A 构建多 agent 框架
- **Query 合成**: 使用 Gyrfalcon 生成多样化的测试查询
- **Trace 收集**: 执行 agents 并捕获执行轨迹
- **数据转换**: 将 traces 转换为训练就绪的格式
- **模块化设计**: 可以独立或组合使用各个组件

---

## 📋 前置要求

- **Python 3.12+** - Python 运行环境
- **UV Package Manager** - 依赖管理工具
- **Git** - 用于克隆仓库和管理 submodules

---

## 🚀 安装

### 1. 安装 UV Package Manager

UV 用于依赖管理，安装方法请参考 [UV 官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

> **提示：** UV 会自动管理 Python 版本，如果您没有 Python 3.12+，UV 会自动安装。

### 2. 克隆仓库

```bash
# 克隆时包含 submodules - 会自动处理嵌套的 submodules
git clone --recursive https://github.com/nex-agi/NexGAP.git
cd NexGAP

# 如果已经克隆，可以后续初始化 submodules
git submodule update --init --recursive

# 安装项目依赖
uv sync
```

### 3. 配置环境

```bash
# 复制示例配置文件
cp NexA4A/.env.example .env

# 编辑 .env 文件，填入您的 API 密钥和配置
# 必需配置：LLM_API_KEY、Langfuse 凭证
```

**最小化 `.env` 配置示例：**

```bash
# LLM 配置 - 必需
LLM_API_KEY=your-api-key
LLM_BASE_URL=your-base-url
LLM_MODEL=your-model

# Langfuse 配置 - trace 收集必需
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

> 完整配置选项请参考 `.env.example` 文件

---

## 🎯 使用

### 基本工作流程

```bash
# 1. 创建 agent framework
cd NexA4A
uv run agent4agent.py interactive
# → 选择 "5. Build a multi-agent framework"
# → 描述您想要的 framework
# → 等待构建完成

# 2. 生成 framework 配置
# → 使用 GyrfalconFrameworkGenerator sub-agent
# → 输入您的 query：包含源路径（/path/to/your/framework）和目标路径（/path/to/Gyrfalcon/frameworks/{framework}）

# 3. 生成测试 queries
cd ../Gyrfalcon
uv run main.py --framework my_framework --num-queries 10 --language english

# 4. 执行并收集 traces
cd ..
uv run run_end_to_end.py \
  --query-filepath Gyrfalcon/output/*_queries_*.jsonl \
  --output-dir output/my_framework \
  --max-workers 5 \
  --max-queries 10

# ✅ 训练数据：output/my_framework/converted_trace/*.jsonl
```

## 📂 项目结构

```
NexGAP/
├── NexA4A/          # Agent 创建框架（submodule）
├── Gyrfalcon/               # Query 合成系统
├── converter/               # Trace 转换工具
├── run_end_to_end.py        # 主执行脚本
└── docs/                    # 文档
    ├── workflow.md          # Pipeline 可视化
    ├── workflow_cn.md       # Pipeline 可视化（中文）
    ├── converter.md         # Converter 使用说明
    ├── converter_cn.md      # Converter 使用说明（中文）
    ├── gyrfalcon.md         # Gyrfalcon 使用说明
    └── gyrfalcon_cn.md      # Gyrfalcon 使用说明（中文）
```

---

## 📚 文档

- **[工作流程指南](workflow_cn.md)** - Pipeline 可视化和架构
- **[Converter 工具](converter_cn.md)** - Trace 检索和转换
- **[Gyrfalcon 使用](gyrfalcon_cn.md)** - Query 合成系统
- **[NexA4A](https://github.com/nex-agi/NexA4A/blob/main/docs/README_cn.md)** - Agent 创建框架

---

## 🤝 贡献

欢迎贡献！请先阅读我们的 [Contributing Guidelines](../CONTRIBUTING.md)。

---

## 📄 许可证

本项目采用 Apache License 2.0 许可 - 详见 [LICENSE](../LICENSE) 文件。

---

## 🔗 链接

- **NexA4A**: [GitHub Repository](https://github.com/nex-agi/NexA4A)
- **Issues**: [Report a bug](https://github.com/nex-agi/NexGAP/issues)
- **Documentation**: [Full docs](./)
