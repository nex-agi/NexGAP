# NexGAP Complete Workflow Diagram

This document provides a visual explanation of the complete NexGAP workflow.

---

## 🔄 End-to-End Workflow

```mermaid
graph TB
    Start([Start]) --> Step1

    subgraph Phase1["Phase 1: Agent Creation"]
        Step1[NexA4A<br/>Create Agent/Framework]
        Step1 --> Output1[created_frameworks/<br/>your_framework/]
    end

    Output1 --> Step2

    subgraph Phase2["Phase 2: Configuration Generation"]
        Step2{Choose Configuration Method}
        Step2 -->|Automatic| Auto[GyrfalconFrameworkGenerator<br/>Auto-generate 4 config files]
        Step2 -->|Manual| Manual[Manual creation<br/>Reference existing frameworks]
        Auto --> Output2
        Manual --> Output2
        Output2[Gyrfalcon/frameworks/<br/>your_framework/<br/>- config.json<br/>- framework_config.yaml<br/>- persona.jsonl<br/>- problem_types.json]
    end

    Output2 --> Step3

    subgraph Phase3["Phase 3: Query Generation"]
        Step3[Gyrfalcon<br/>Generate test queries]
        Step3 --> Output3[output/<br/>your_framework_queries.jsonl]
    end

    Output3 --> Step4

    subgraph Phase4["Phase 4: Execution and Collection"]
        Step4[run_end_to_end.py<br/>Execute queries + Collect traces]
        Step4 --> Output4A[langfuse_trace/<br/>Raw Langfuse trace]
        Step4 --> Output4B[converted_trace/<br/>ChatCompletion format<br/>NexAU XML or OpenAI tool call]
    end

    Output4B --> End([Complete<br/>Training data ready])

    Note5["💡 converted_trace/ is already<br/>in ChatCompletion format<br/>Ready for training"]

    Output4B -.-> Note5

    style Start fill:#e1f5e1
    style End fill:#e1f5e1
    style Step1 fill:#fff4e6
    style Step2 fill:#fff4e6
    style Step3 fill:#e6f3ff
    style Step4 fill:#f0e6ff
    style Output4B fill:#ffcccc
    style Note5 fill:#ffffcc
```

---

## 📊 Data Flow Details

### Phase 1: Agent Creation

**Tool**: NexA4A

```mermaid
graph LR
    User[User Requirements] --> A4A[Agent4Agent]
    A4A --> Prompt[Generate System Prompts]
    A4A --> Tools[Select/Create Tools]
    A4A --> Config[Generate Config Files]

    Prompt --> Framework[Complete Framework]
    Tools --> Framework
    Config --> Framework

    Framework --> Storage[src/created_frameworks/<br/>framework_name/]

    style User fill:#e1f5e1
    style Framework fill:#fff4e6
    style Storage fill:#e6f3ff
```

**Input**:
- User requirement description (natural language)

**Output**:
```
src/created_frameworks/your_framework/
├── your_framework.yaml          # Framework configuration
├── agents/                      # Agent definitions
│   ├── agent1.yaml
│   └── agent2.yaml
└── tools/                       # Tool implementations (if custom)
    └── custom_tool.py
```

---

### Phase 2: Configuration Generation

**Tool**: GyrfalconFrameworkGenerator (recommended) or manual creation

```mermaid
graph TB
    Source[NexA4A Framework] --> Gen{Configuration Generation Method}

    Gen -->|Automatic| GFG[GyrfalconFrameworkGenerator]
    Gen -->|Manual| Manual[Manual Editing]

    GFG --> Config1[config.json<br/>Framework, agents, tools]
    GFG --> Config2[framework_config.yaml<br/>Metadata, descriptions]
    GFG --> Config3[persona.jsonl<br/>500 personas]
    GFG --> Config4[problem_types.json<br/>Problem type tree]

    Manual --> Config1
    Manual --> Config2
    Manual --> Config3
    Manual --> Config4

    Config1 --> Output[Gyrfalcon/frameworks/<br/>your_framework/]
    Config2 --> Output
    Config3 --> Output
    Config4 --> Output

    style Source fill:#fff4e6
    style GFG fill:#e6f3ff
    style Output fill:#f0e6ff
```

**4 Configuration Files Explained**:

1. **config.json** - Framework core configuration
```json
{
  "framework_name": "...",
  "agents": [...],  // Agent definitions
  "tools": [...]    // Tool definitions
}
```

2. **framework_config.yaml** - Metadata
```yaml
framework:
  name: ...
  description_en: ...
  description_zh: ...
  tags: [...]
```

3. **persona.jsonl** - User personas (500 entries)
```jsonl
{"persona_en": "...", "persona_zh": "..."}
{"persona_en": "...", "persona_zh": "..."}
```

4. **problem_types.json** - Problem type tree
```json
{
  "root": {
    "children": {
      "Type1": {
        "weight": 1.0,
        "children": {...}
      }
    }
  }
}
```

---

### Phase 3: Query Generation

**Tool**: Gyrfalcon

```mermaid
graph TB
    Config[4 config files] --> Pipeline[Gyrfalcon Pipeline]

    Pipeline --> Sample[Problem type sampling<br/>Based on problem_types.json]
    Sample --> Persona[Persona sampling<br/>From persona.jsonl]
    Persona --> Difficulty[Difficulty assignment<br/>easy/medium/hard]
    Difficulty --> LLM[LLM generates query<br/>Based on framework_config.yaml]

    LLM --> Optional{Optional Enhancements}
    Optional -->|websearch| Web[Web Search]
    Optional -->|fuzzify| Fuzz[Query Fuzzification]
    Optional -->|url| URL[URL Processing]

    Web --> Output[queries.jsonl]
    Fuzz --> Output
    URL --> Output
    LLM --> Output

    Output --> Stats[Statistics.json]
    Output --> Viz[Visualization.html]

    style Pipeline fill:#e6f3ff
    style LLM fill:#fff4e6
    style Output fill:#f0e6ff
```

**Output Format** (`queries.jsonl`):
```json
{
  "content": "How to diagnose hypertension?",
  "difficulty": "medium",
  "context": {
    "problem_type": "Disease Diagnosis",
    "persona": "Family Doctor",
    "framework": "medical_diagnostic_assistant"
  },
  "metadata": {
    "generated_at": "2025-11-14T12:00:00",
    "used_web_search": false,
    "fuzzified": false
  }
}
```

**Performance**:
- Serial (1 worker): ~10 queries/min
- Parallel (4 workers): ~34 queries/min (3.4x)
- Parallel (8 workers): ~60 queries/min (6x)

---

### Phase 4: Execution and Collection

**Tool**: run_end_to_end.py

```mermaid
graph TB
    Queries[queries.jsonl] --> Parallel[Parallel Processing<br/>max-workers controls concurrency]

    Parallel --> Execute[NexA4A<br/>Execute query]
    Execute --> Capture[Capture trace_id]
    Capture --> Wait[Wait for Langfuse recording]
    Wait --> Fetch[Fetch trace from Langfuse]

    Fetch --> Save1[Save raw trace<br/>langfuse_trace/]
    Save1 --> Convert{Convert Format}

    Convert -->|Default| NexAU[NexAU XML<br/>converted_trace/]
    Convert -->|--use-openai-format| OpenAI[OpenAI tool call<br/>converted_trace/]

    NexAU --> Filter[XML Validation Filter]
    Filter --> ToolFormat{--tool-call-format?}

    ToolFormat -->|Yes| SpecFormat[Qwen/GLM/...-like]
    ToolFormat -->|No| Done1[Complete]

    OpenAI --> Done2[Complete]
    SpecFormat --> Done3[Complete]

    Done1 --> Next{More queries?}
    Done2 --> Next
    Done3 --> Next

    Next -->|Yes| Parallel
    Next -->|No| Summary[Generate statistics report]

    style Execute fill:#fff4e6
    style Fetch fill:#e6f3ff
    style Convert fill:#f0e6ff
    style Summary fill:#e1f5e1
```

**Parallel Processing**: Use `--max-workers` to control concurrency
- Default: 3 workers
- Recommended: 5-10 workers (depending on system resources)
- Note: No resume from interruption, restart required after interruption

**Output Structure**:
```
output/
├── langfuse_trace/          # Raw Langfuse spans
│   └── framework_traceid_timestamp.jsonl
├── converted_trace/         # ChatCompletion format (default NexAU XML)
│   └── framework_traceid_timestamp_chatcompletion.jsonl
│   └── framework_traceid_timestamp_chatcompletion_qwen.jsonl  # If --tool-call-format specified
└── logs/                    # Execution log for each query
    └── query_001_framework_timestamp.log
```

---

### Phase 5: Training Data Ready

**Output**: converted_trace/ (ChatCompletion format)

```mermaid
graph TB
    Input[converted_trace/<br/>ChatCompletion format] --> Ready[✅ Training Data Ready]

    Ready --> Use1[SFT Training]
    Ready --> Use2[Evaluation Testing]

    style Input fill:#ffcccc
    style Ready fill:#e1f5e1
    style Use1 fill:#e6f3ff
    style Use2 fill:#e6f3ff
```

**Final Training Data Format**:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "How to diagnose hypertension?"
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "search_medical_database",
            "arguments": "{\"query\": \"hypertension diagnostic criteria\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "Hypertension diagnostic criteria: Systolic BP ≥140mmHg or Diastolic BP ≥90mmHg..."
    },
    {
      "role": "assistant",
      "content": "According to medical database query results, hypertension diagnostic criteria are..."
    }
  ],
  "metadata": {
    "trace_id": "cm4a5b2c1...",
    "framework": "medical_diagnostic_assistant",
    "difficulty": "medium",
    "quality_score": 0.95
  }
}
```

---

## 🔀 Alternative Workflows

### Workflow A: Minimal Flow (Test Single Query)

```mermaid
graph LR
    A[NexA4A] --> B[Manual Input Query]
    B --> C[View Output]

    style A fill:#fff4e6
    style B fill:#e6f3ff
    style C fill:#e1f5e1
```

**Use Case**: Quick test of agent functionality

---

### Workflow B: Skip Agent Creation (Use Existing Framework)

```mermaid
graph LR
    A[Existing Framework] --> B[Generate Configuration]
    B --> C[Gyrfalcon]
    C --> D[run_end_to_end]
    D --> E[Converter]

    style A fill:#e6f3ff
    style C fill:#fff4e6
    style E fill:#ffcccc
```

**Use Case**: Already have framework, only need to generate training data

---

### Workflow C: Batch Process Multiple Frameworks

```mermaid
graph TB
    Start[Framework List] --> Loop{Iterate Frameworks}

    Loop --> Gen[Generate Configuration]
    Gen --> Queries[Generate Queries]
    Queries --> Execute[Execute Collection]
    Execute --> Convert[Convert Data]

    Convert --> Loop
    Loop --> Merge[Merge All Data]
    Merge --> End[Unified Training Dataset]

    style Start fill:#e1f5e1
    style End fill:#ffcccc
```

**Use Case**: Large-scale data generation

---

## 📈 Performance Considerations

### Parallelization Strategy

| Phase | Tool | Parallelization | Recommended Config |
|------|------|--------|----------|
| 1. Agent Creation | NexA4A | ❌ Serial | - |
| 2. Config Generation | GyrfalconFrameworkGenerator | ❌ Serial | - |
| 3. Query Generation | Gyrfalcon | ✅ Multi-process | 4-8 workers |
| 4. Execution Collection | run_end_to_end | ✅ Multi-thread | 3-10 workers (--max-workers) |
| 5. Format Conversion | Automatic (embedded in run_end_to_end) | ✅ Automatic | No configuration needed |

### Resource Consumption Estimate

**Complete workflow for generating 1000 queries**:
- **Time**: ~2-3 hours (including execution)
- **API Calls**:
  - Gyrfalcon: ~1000 LLM calls
  - Agent execution: ~1000-5000 (depending on agent complexity)
- **Storage**: ~500MB - 2GB (depending on trace size)
- **Cost**: $10-50 (depending on model and token usage)

---

## 🛠️ Fault Recovery

### Fault Tolerance Mechanism for Each Phase

```mermaid
graph TB
    Start[Start Execution] --> Execute[Parallel Execute Tasks]

    Execute --> Success{Success?}

    Success -->|Yes| Log[Record to Log]
    Success -->|No| Retry{Can Retry?}

    Retry -->|Network Error| Execute
    Retry -->|Other| Mark[Mark as Failed]

    Log --> Next{More Tasks?}
    Mark --> Next

    Next -->|Yes| Execute
    Next -->|No| Done[Complete<br/>Display Statistics]

    style Log fill:#e1f5e1
    style Mark fill:#ffcccc
    style Done fill:#e6f3ff
```

**Fault Tolerance by Phase**:
- ✅ **Gyrfalcon**: Failed query generation will automatically retry
- ⚠️ **run_end_to_end**: No automatic resume from interruption, need to restart after interruption
  - Suggestion: Use `--max-workers` to accelerate execution
  - Suggestion: Start with `--max-queries 10` for testing
- ✅ **Converter**: Automatically integrated, XML validation filters error records

**View Logs**:
```bash
# View all logs in real-time
tail -f output/*/logs/*.log

# View log for specific query
cat output/framework/logs/query_001_framework_timestamp.log
```
