# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 常用命令速查

# 1. 串行生成 20 条英文查询
python main.py \
  --framework research_orchestrator \
  --num-queries 20 \
  --language english

# 2. 并行生成 200 条中文查询（4 个 worker）
python main.py \
  --framework research_orchestrator \
  --num-queries 200 \
  --num-workers 4 \
  --language chinese \
  --output-dir output/research_cn

# 3. 自定义难度分布
python main.py \
  --framework research_orchestrator \
  --num-queries 100 \
  --difficulty-distribution "easy:0.2,medium:0.5,hard:0.3"

# 4. 启用 Web 检索（40% 概率，每次最多 5 条结果）
python main.py \
  --framework research_orchestrator \
  --num-queries 100 \
  --websearch-prob 0.4 \
  --websearch-max-results 5

# 5. 模糊化后处理：30% 概率触发子代理
python main.py \
  --framework research_orchestrator \
  --num-queries 60 \
  --fuzzify-prob 0.3

# 6. 提升问题类型扩展概率至 20%
python main.py \
  --framework research_orchestrator \
  --num-queries 100 \
  --problem-type-expand-prob 0.2

# 7. Debug 模式：只保存 prompt
python main.py \
  --framework research_orchestrator \
  --num-queries 1 \
  --debug-prompts

# 8. 批量配置示例
python main.py --batch-config configs/research_batch.json

# 9. 单独测试 Web 搜索脚本（随机 persona + problem type）
SERPER_API_KEY=xxxx \
python scripts/test_serper_search.py --limit 5

# 10. 指定全部主要参数的示例命令
python main.py \
  --framework consignal_cn \
  --num-queries 500 \
  --num-workers 100 \
  --language chinese \
  --export-format jsonl \
  --difficulty-distribution "easy:0.1,medium:0.4,hard:0.5" \
  --websearch-prob 0.3 \
  --websearch-max-results 10 \
  --problem-type-expand-prob 0.01 \
  --enable-url-processing \
  --fuzzify-prob 0.0 \
  --log-level INFO \
  --include-framework-description \


# 11. 启用文件需求分析和文件下载功能
python main.py \
  --framework DataAnalysisOrchestratorAgent \
  --num-queries 50 \
  --enable-file-analysis

# 12. 启用URL处理功能（提取、验证、修复查询中的URL）
python main.py \
  --framework DataAnalysisOrchestratorAgent \
  --num-queries 50 \
  --language chinese \
  --enable-url-processing

# 13. 同时启用文件分析和URL处理
python main.py \
  --framework DataAnalysisOrchestratorAgent \
  --num-queries 100 \
  --num-workers 4 \
  --language chinese \
  --enable-file-analysis \
  --enable-url-processing

# 14. 对已有查询 JSONL 批量进行模糊化
python scripts/fuzzify_queries.py \
  --input test/querys_zh.jsonl \
  --output test/querys_zh_fuzzy_llm.jsonl
