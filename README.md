# NexGAP

[English](#) | [中文](docs/README_cn.md)

**General Agentic Data Pipeline**

An end-to-end pipeline for generating high-quality agentic training data. Built on NexAU agent framework and NexA4A's agent-building capabilities, covering agent creation, problem synthesis, trajectory generation, and trace processing.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## ✨ Features

- **Agent Creation**: Build multi-agent frameworks with NexA4A
- **Query Synthesis**: Generate diverse test queries with Gyrfalcon
- **Trace Collection**: Execute agents and capture execution traces
- **Data Conversion**: Transform traces into training-ready formats
- **Modular Design**: Use components independently or together

---

## 📋 Prerequisites

- **Python 3.12+**
- **UV Package Manager**
- **Git** (for submodule management)

---

## 🚀 Installation

### 1. Install UV Package Manager

UV is required for dependency management. See [UV installation guide](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

> **Note:** UV automatically manages Python versions. If you don't have Python 3.12+, UV will install it for you.

### 2. Clone Repository

```bash
# Clone with submodules (includes nested submodules)
git clone --recursive https://github.com/nex-agi/NexGAP.git
cd NexGAP

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Install dependencies
uv sync
```

### 3. Configure Environment

```bash
# Copy example config
cp NexA4A/.env.example .env

# Edit .env with your settings
# Required: LLM_API_KEY, LANGFUSE credentials
```

**Minimal `.env` configuration:**

```bash
# LLM Configuration (Required)
LLM_API_KEY=your-api-key
LLM_BASE_URL=your-base-url
LLM_MODEL=your-model

# Langfuse (Required for trace collection)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

> See `.env.example` for all configuration options

---

## 🎯 Usage

### Basic Workflow

```bash
# 1. Create agent framework
cd NexA4A
uv run agent4agent.py interactive
# → Select "5. Build a multi-agent framework"
# → Describe the framework you want
# → Wait util build done

# 2. Generate framework config
# → Use GyrfalconFrameworkGenerator sub-agent
# → Input you query: containing the source path(/path/to/your/framework), target path(/path/to/Gyrfalcon/frameworks/{framework})

# 3. Generate test queries
cd ../Gyrfalcon
uv run main.py --framework my_framework --num-queries 10 --language english

# 4. Execute and collect traces
cd ..
uv run run_end_to_end.py \
  --query-filepath Gyrfalcon/output/*_queries_*.jsonl \
  --output-dir output/my_framework \
  --max-workers 5 \
  --max-queries 10

# ✅ Training data: output/my_framework/converted_trace/*.jsonl
```

## 📂 Project Structure

```
NexGAP/
├── NexA4A/          # Agent creation framework (submodule)
├── Gyrfalcon/               # Query synthesis system
├── converter/               # Trace conversion tools
├── run_end_to_end.py        # Main execution script
└── docs/                    # Documentation
    ├── README_cn.md         # Chinese documentation
    ├── workflow.md          # Pipeline visualization
    ├── workflow_cn.md       # Pipeline visualization (Chinese)
    ├── converter.md         # Converter usage
    ├── converter_cn.md      # Converter usage (Chinese)
    ├── gyrfalcon.md         # Gyrfalcon usage
    └── gyrfalcon_cn.md      # Gyrfalcon usage (Chinese)
```

---

## 📚 Documentation

- **[Workflow Guide](docs/workflow.md)** - Pipeline visualization and architecture
- **[Converter Tools](docs/converter.md)** - Trace retrieval and conversion
- **[Gyrfalcon Usage](docs/gyrfalcon.md)** - Query synthesis system
- **[NexA4A](https://github.com/nex-agi/NexA4A/blob/main/README.md)** - Agent creation framework

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **NexA4A**: [GitHub Repository](https://github.com/nex-agi/NexA4A)
- **Issues**: [Report a bug](https://github.com/nex-agi/NexGAP/issues)
- **Documentation**: [Full docs](docs/)
