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

"""
Path Sampling Statistics and New Tag Generation

Manages sampling statistics for problem type paths and generates new tags
with small probability to expand the tree dynamically.
"""

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PathSamplingStats:
    """
    Tracks sampling statistics for problem type paths.
    Enables weighted sampling that prioritizes less-sampled paths.

    Thread-safe for multi-process execution using file-based locking.
    """

    def __init__(self, framework_name: str, stats_dir: str = "./frameworks"):
        self.framework_name = framework_name
        self.stats_dir = Path(stats_dir)
        self.stats_file = self.stats_dir / framework_name / "tag_sampling_stats.json"

        # path_id -> sampling_count (loaded from file)
        self.path_counts: Dict[str, int] = defaultdict(int)

        # path_id -> delta_count (new samples recorded since last load/save)
        self.delta_counts: Dict[str, int] = defaultdict(int)

        # Statistics
        self.total_samples = 0
        self.new_samples = 0  # Samples recorded since last load/save
        self.last_updated = time.time()

        # Load existing stats
        self._load_stats()

    def _load_stats(self):
        """Load sampling statistics from file"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.path_counts = defaultdict(int, data.get("path_counts", {}))
                self.total_samples = data.get("total_samples", 0)
                self.last_updated = data.get("last_updated", time.time())

                # Reset delta counts after loading
                self.delta_counts = defaultdict(int)
                self.new_samples = 0

                logger.info(
                    f"Loaded sampling stats for {self.framework_name}: "
                    f"{len(self.path_counts)} paths, {self.total_samples} total samples"
                )
            except Exception as e:
                logger.warning(f"Failed to load sampling stats: {e}. Starting fresh.")
                self.path_counts = defaultdict(int)
                self.delta_counts = defaultdict(int)
                self.total_samples = 0
                self.new_samples = 0

    def save_stats(self):
        """
        Save sampling statistics to file with proper locking for multi-process safety.

        In parallel execution, multiple workers may try to save stats simultaneously.
        This method:
        1. Acquires exclusive lock on stats file
        2. Reloads current stats from file
        3. Merges delta counts (new samples since last load) with loaded stats
        4. Saves merged result
        5. Resets delta counts
        """
        try:
            # Import lock here to avoid circular import
            from .file_lock import sampling_stats_lock

            # Acquire lock for stats file modification
            with sampling_stats_lock(
                self.framework_name, str(self.stats_dir), timeout=10.0
            ):
                # Reload stats from file to get latest from other processes
                if self.stats_file.exists():
                    try:
                        with open(self.stats_file, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Load current file state
                        file_path_counts = defaultdict(int, data.get("path_counts", {}))
                        file_total_samples = data.get("total_samples", 0)
                    except Exception as e:
                        logger.warning(f"Failed to reload stats for merging: {e}")
                        file_path_counts = defaultdict(int)
                        file_total_samples = 0
                else:
                    file_path_counts = defaultdict(int)
                    file_total_samples = 0

                # Merge delta counts into file counts
                merged_counts = file_path_counts.copy()
                for path_id, delta in self.delta_counts.items():
                    merged_counts[path_id] += delta

                # Update total samples
                merged_total = file_total_samples + self.new_samples

                # Prepare data to save
                self.stats_file.parent.mkdir(exist_ok=True, parents=True)

                data = {
                    "framework": self.framework_name,
                    "path_counts": dict(merged_counts),
                    "total_samples": merged_total,
                    "last_updated": time.time(),
                    "statistics": {
                        "total_paths_tracked": len(merged_counts),
                        "min_count": (
                            min(merged_counts.values()) if merged_counts else 0
                        ),
                        "max_count": (
                            max(merged_counts.values()) if merged_counts else 0
                        ),
                        "avg_count": (
                            sum(merged_counts.values()) / len(merged_counts)
                            if merged_counts
                            else 0
                        ),
                    },
                }

                # Save merged stats
                with open(self.stats_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # Update our in-memory state to match what was saved
                self.path_counts = merged_counts
                self.total_samples = merged_total

                # Reset delta counts after successful save
                self.delta_counts = defaultdict(int)
                self.new_samples = 0

                logger.debug(
                    f"Saved sampling stats for {self.framework_name}: "
                    f"{len(merged_counts)} paths, {merged_total} total samples"
                )

        except TimeoutError as e:
            logger.warning(
                f"Failed to acquire sampling stats lock: {e}. Stats not saved."
            )
        except Exception as e:
            logger.error(f"Failed to save sampling stats: {e}")

    def record_sample(self, path_id: str):
        """
        Record a sample for a given path.

        This updates both the in-memory counts and delta counts.
        Delta counts track new samples since last save for proper merging in parallel execution.
        """
        self.path_counts[path_id] += 1
        self.delta_counts[path_id] += 1
        self.total_samples += 1
        self.new_samples += 1
        self.last_updated = time.time()

    def get_path_weight(self, path_id: str, all_path_ids: List[str]) -> float:
        """
        Calculate sampling weight for a path based on inverse frequency.
        Less sampled paths get higher weights.

        Uses formula: weight = 1 / (count + 1)^alpha
        where alpha controls how strongly we prefer less-sampled paths
        """
        count = self.path_counts.get(path_id, 0)
        alpha = 1.5  # Tunable parameter: higher = stronger preference for less-sampled

        # Add 1 to avoid division by zero
        weight = 1.0 / ((count + 1) ** alpha)
        return weight

    def get_weighted_path_probabilities(self, all_path_ids: List[str]) -> List[float]:
        """
        Get normalized probability distribution for all paths.
        Less sampled paths have higher probability.
        """
        weights = [self.get_path_weight(pid, all_path_ids) for pid in all_path_ids]
        total_weight = sum(weights)

        if total_weight == 0:
            # Uniform distribution if no weights
            return [1.0 / len(all_path_ids)] * len(all_path_ids)

        probabilities = [w / total_weight for w in weights]
        return probabilities

    def sample_path_index(self, all_path_ids: List[str]) -> int:
        """
        Sample a path index using weighted sampling.
        Returns the index of the selected path.
        """
        probabilities = self.get_weighted_path_probabilities(all_path_ids)
        return random.choices(range(len(all_path_ids)), weights=probabilities, k=1)[0]


class NewTagGenerator:
    """
    Generates new problem type tags to expand the tree dynamically.
    Uses LLM to create contextually relevant but distinct new problem types.
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient instance for generating new tags
        """
        self.llm_client = llm_client

        # Probability of generating a new tag
        # For 2,000 samples: 0.1 = ~200 new tags
        # For 5,000 samples: 0.1 = ~500 new tags
        # For 10,000 samples: 0.1 = ~1000 new tags
        self.new_tag_probability = 0.1  # 10% chance

    def should_generate_new_tag(self) -> bool:
        """Decide whether to generate a new tag based on probability"""
        return random.random() < self.new_tag_probability

    def _format_framework_capabilities(
        self, framework_config, language: str = "english"
    ) -> str:
        """
        Format framework capabilities into a readable string for the prompt.

        Args:
            framework_config: FrameworkConfig object from framework_manager
            language: Language for formatting ("english" or "chinese")

        Returns:
            Formatted string describing framework capabilities
        """
        if not framework_config:
            return ""

        lines = []

        if language.lower() == "chinese":
            # Framework description
            lines.append(f"框架总体能力：{framework_config.description}")
            lines.append("")

            # Subagents
            if framework_config.subagents:
                lines.append("可用的智能体：")
                for agent in framework_config.subagents:
                    agent_name = agent.get("name", "未命名")
                    agent_desc = agent.get("description", "")
                    capabilities = agent.get("capabilities", [])
                    lines.append(f"  - {agent_name}: {agent_desc}")
                    if capabilities:
                        cap_str = ", ".join(capabilities)
                        lines.append(f"    能力: {cap_str}")
                lines.append("")

            # Tools
            if framework_config.tools:
                lines.append("可用的工具：")
                for tool in framework_config.tools:
                    tool_name = tool.get("name", "未命名")
                    tool_desc = tool.get("description", "")
                    capabilities = tool.get("capabilities", [])
                    lines.append(f"  - {tool_name}: {tool_desc}")
                    if capabilities:
                        cap_str = ", ".join(capabilities)
                        lines.append(f"    能力: {cap_str}")
        else:
            # Framework description
            lines.append(
                f"Framework Overall Capabilities: {framework_config.description}"
            )
            lines.append("")

            # Subagents
            if framework_config.subagents:
                lines.append("Available Agents:")
                for agent in framework_config.subagents:
                    agent_name = agent.get("name", "unnamed")
                    agent_desc = agent.get("description", "")
                    capabilities = agent.get("capabilities", [])
                    lines.append(f"  - {agent_name}: {agent_desc}")
                    if capabilities:
                        cap_str = ", ".join(capabilities)
                        lines.append(f"    Capabilities: {cap_str}")
                lines.append("")

            # Tools
            if framework_config.tools:
                lines.append("Available Tools:")
                for tool in framework_config.tools:
                    tool_name = tool.get("name", "unnamed")
                    tool_desc = tool.get("description", "")
                    capabilities = tool.get("capabilities", [])
                    lines.append(f"  - {tool_name}: {tool_desc}")
                    if capabilities:
                        cap_str = ", ".join(capabilities)
                        lines.append(f"    Capabilities: {cap_str}")

        return "\n".join(lines)

    def generate_new_tag(
        self,
        parent_node,
        existing_siblings: List[Any],
        framework_config=None,
        language: str = "english",
    ) -> Optional[Dict[str, str]]:
        """
        Generate a new problem type tag as a child of parent_node.

        Args:
            parent_node: The parent ProblemTypeNode
            existing_siblings: List of existing sibling nodes
            framework_config: FrameworkConfig object for capability constraints
            language: Language for generation

        Returns:
            Dict with 'id', 'en', 'zh' keys, or None if generation fails
        """
        try:
            # Build sibling context
            sibling_labels_en = [s.en for s in existing_siblings]
            sibling_labels_zh = [s.zh for s in existing_siblings]

            # Generate prompt
            prompt = self._build_new_tag_prompt(
                parent_label_en=parent_node.en,
                parent_label_zh=parent_node.zh,
                sibling_labels_en=sibling_labels_en,
                sibling_labels_zh=sibling_labels_zh,
                framework_config=framework_config,
                language=language,
            )

            # Call LLM
            response = self.llm_client.generate_completion(prompt)

            # Parse response
            new_tag = self._parse_new_tag_response(response, parent_node.id)

            if new_tag:
                logger.info(
                    f"Generated new tag: {new_tag['en']} (parent: {parent_node.en})"
                )

            return new_tag

        except Exception as e:
            logger.error(f"Failed to generate new tag: {e}")
            return None

    def _build_new_tag_prompt(
        self,
        parent_label_en: str,
        parent_label_zh: str,
        sibling_labels_en: List[str],
        sibling_labels_zh: List[str],
        framework_config,
        language: str,
    ) -> str:
        """Build prompt for generating new problem type"""

        # Format framework capabilities
        framework_capabilities = self._format_framework_capabilities(
            framework_config, language
        )

        if language.lower() == "chinese":
            prompt = f"""你是一个问题类型分类专家。请基于以下上下文生成一个新的问题类型。

父类别（中文）：{parent_label_zh}
父类别（英文）：{parent_label_en}

已有的兄弟类别（中文）：
{chr(10).join(f"- {label}" for label in sibling_labels_zh)}

已有的兄弟类别（英文）：
{chr(10).join(f"- {label}" for label in sibling_labels_en)}

框架能力约束：
{framework_capabilities}

要求：
1. 新问题类型必须是父类别的子类型
2. 新问题类型必须与所有已有兄弟类别不同，不能重复或交叉
3. 新问题类型必须有明确的界限和定义
4. **新问题类型必须能够被框架的现有能力解决（基于上述框架能力约束）**
5. 提供中英文两种表述

请严格按照以下JSON格式输出（不要包含任何其他文字）：
{{
  "en": "English problem type name",
  "zh": "中文问题类型名称",
  "id": "snake_case_identifier"
}}"""
        else:
            prompt = f"""You are a problem type taxonomy expert. Generate a new problem type based on the following context.

Parent Category (English): {parent_label_en}
Parent Category (Chinese): {parent_label_zh}

Existing Sibling Categories (English):
{chr(10).join(f"- {label}" for label in sibling_labels_en)}

Existing Sibling Categories (Chinese):
{chr(10).join(f"- {label}" for label in sibling_labels_zh)}

Framework Capability Constraints:
{framework_capabilities}

Requirements:
1. The new problem type MUST be a subcategory of the parent
2. The new problem type MUST be different from all existing siblings - no overlap or duplication
3. The new problem type MUST have clear boundaries and definition
4. **The new problem type MUST be solvable by the framework's existing capabilities (based on the framework capability constraints above)**
5. Provide both English and Chinese labels

Output STRICTLY in the following JSON format (no other text):
{{
  "en": "English problem type name",
  "zh": "中文问题类型名称",
  "id": "snake_case_identifier"
}}"""

        return prompt

    def _parse_new_tag_response(
        self, response: str, parent_id: str
    ) -> Optional[Dict[str, str]]:
        """Parse LLM response to extract new tag information"""
        try:
            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in new tag response")
                return None

            data = json.loads(json_match.group())

            # Validate required fields
            if "en" not in data or "zh" not in data or "id" not in data:
                logger.warning("Missing required fields in new tag response")
                return None

            # Ensure ID is unique by adding parent prefix
            data["id"] = f"{parent_id}_{data['id']}"

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse new tag JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing new tag response: {e}")
            return None
