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
URL Repair Agent for Gyrfalcon v5

Suggests alternative URLs using LLM analysis of broken URLs.
"""

import json
import logging
from typing import Any, Dict, List

from .base import Agent, AgentContext, AgentOutput

logger = logging.getLogger(__name__)


class URLRepairAgent(Agent):
    """
    Analyzes broken URLs and suggests alternative working URLs using LLM.
    """

    def __init__(self, llm_client, name: str = "URLRepairAgent"):
        super().__init__(name)
        self.llm_client = llm_client

    def run(self, context: AgentContext) -> AgentOutput:
        """
        Generate repair suggestions for broken URLs.

        Expected context keys:
            - broken_urls: List[Dict] (URLs that failed validation)
            - language: str (optional, "english" or "chinese")

        Adds to context:
            - repair_suggestions: Dict[str, Dict] (suggestions for each broken URL)
        """
        broken_urls = context.get("broken_urls", [])
        language = context.get("language", "english")

        if not broken_urls:
            logger.info("No broken URLs to repair")
            return AgentOutput(success=True, data={"repair_suggestions": {}})

        try:
            logger.info(
                f"Generating repair suggestions for {len(broken_urls)} URL(s)..."
            )

            repair_suggestions = {}

            for url_info in broken_urls:
                original_url = url_info.get("url")
                error = url_info.get("error", "Unknown error")
                context_text = url_info.get("context", "")

                suggestion = self._repair_url(
                    original_url, error, context_text, language
                )
                repair_suggestions[original_url] = suggestion

            return AgentOutput(
                success=True, data={"repair_suggestions": repair_suggestions}
            )

        except Exception as e:
            logger.error(f"URL repair failed: {e}")
            return AgentOutput(success=False, errors=[f"URL repair error: {str(e)}"])

    def _repair_url(
        self, url: str, error: str, context: str, language: str
    ) -> Dict[str, Any]:
        """Generate repair suggestions for a single URL"""

        prompt = self._build_repair_prompt(url, error, context, language)

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=8192,  # Increased for detailed URL analysis with multiple suggestions
            )

            content = response.choices[0].message.content
            logger.debug(
                f"URL repair response (first 500 chars): {content[:500] if content else 'EMPTY'}..."
            )
            return self._parse_response(content, url)

        except Exception as e:
            logger.error(f"Failed to get repair suggestions for {url}: {e}")
            return {
                "repairable": False,
                "suggested_urls": [],
                "reasoning": f"Error: {str(e)}",
                "action": "remove",
                "original_url": url,
            }

    def _build_repair_prompt(
        self, url: str, error: str, context: str, language: str
    ) -> str:
        """Build the LLM prompt for URL repair"""

        if language.lower() == "chinese":
            return f"""你是URL修复专家。分析这个404错误的URL并提供修复方案。

原始URL: {url}
错误: {error}
用途: {context}

分析要点：
1. URL结构拆解：找出可能过期的部分（年份2019?分支master?路径?）
2. 常见修复：master→main, 年份更新, http→https, 域名迁移
3. 数据集搜索：检查Kaggle/UCI/GitHub是否有同名数据

**必须生成3-5个候选URL**，按可能性排序。

严格返回JSON：
{{
  "repairable": true,
  "suggested_urls": ["URL1", "URL2", "URL3", "URL4", "URL5"],
  "reasoning": "说明：URL哪里有问题 + 每个建议的修复思路",
  "action": "repair"
}}

如果确实无法修复才返回：{{"repairable": false, "suggested_urls": [], "action": "remove"}}

请直接返回JSON，不要额外说明。
"""
        else:
            return f"""You are a URL repair expert. Analyze this broken URL and provide repair solutions.

Original URL: {url}
Error: {error}
Context: {context}

Analysis points:
1. URL structure breakdown: identify potentially outdated parts (year 2019? branch master? path?)
2. Common fixes: master→main, year updates, http→https, domain migration
3. Dataset search: check if Kaggle/UCI/GitHub has the same dataset

**Must generate 3-5 candidate URLs**, ordered by probability.

Return STRICT JSON:
{{
  "repairable": true,
  "suggested_urls": ["URL1", "URL2", "URL3", "URL4", "URL5"],
  "reasoning": "Explanation: what's wrong with the URL + repair approach for each suggestion",
  "action": "repair"
}}

Only return {{"repairable": false, "suggested_urls": [], "action": "remove"}} if truly irreparable.

Return ONLY JSON, no extra explanation.
"""

    def _parse_response(self, response: str, original_url: str) -> Dict[str, Any]:
        """Parse the LLM response"""

        try:
            # Extract JSON from response
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)

            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("Response is not a dictionary")

            if "repairable" not in data:
                data["repairable"] = False

            if "suggested_urls" not in data:
                data["suggested_urls"] = []

            if "action" not in data:
                data["action"] = "remove" if not data["repairable"] else "repair"

            if "reasoning" not in data:
                data["reasoning"] = "No reasoning provided"

            # Ensure suggested_urls is a list
            if not isinstance(data["suggested_urls"], list):
                data["suggested_urls"] = []

            # Check consistency
            if data.get("repairable") and len(data.get("suggested_urls", [])) == 0:
                logger.warning("Marked as repairable but no suggested URLs provided")
                data["repairable"] = False
                data["action"] = "remove"

            data["original_url"] = original_url

            return data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse repair response as JSON: {e}")
            logger.error(f"Response: {response}")
            return {
                "repairable": False,
                "suggested_urls": [],
                "reasoning": f"JSON parse error: {str(e)}",
                "action": "remove",
                "original_url": original_url,
            }
        except Exception as e:
            logger.error(f"Error parsing repair response: {e}")
            return {
                "repairable": False,
                "suggested_urls": [],
                "reasoning": f"Parse error: {str(e)}",
                "action": "remove",
                "original_url": original_url,
            }
