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
Query Rewrite Agent for URL Processing

Rewrites queries to use fixed URLs or remove broken URLs.
"""

import json
import logging
from typing import Any, Dict, List

from .base import Agent, AgentContext, AgentOutput

logger = logging.getLogger(__name__)


class URLQueryRewriteAgent(Agent):
    """
    Rewrites queries with fixed or removed URLs.
    """

    def __init__(self, llm_client, name: str = "URLQueryRewriteAgent"):
        super().__init__(name)
        self.llm_client = llm_client

    def run(self, context: AgentContext) -> AgentOutput:
        """
        Rewrite query with URL changes.

        Expected context keys:
            - query: str (original query)
            - url_changes: List[Dict] (URL replacements/removals)
            - language: str (optional, "english" or "chinese")

        Adds to context:
            - rewritten_query: str (query with URL changes applied)
            - query_was_rewritten: bool (whether rewriting occurred)
        """
        query = context.get("query")
        url_changes = context.get("url_changes", [])
        language = context.get("language", "english")

        if not query:
            return AgentOutput(
                success=False, errors=["No query provided for rewriting"]
            )

        if not url_changes:
            logger.info("No URL changes to apply")
            return AgentOutput(
                success=True,
                data={"rewritten_query": query, "query_was_rewritten": False},
            )

        try:
            logger.info(f"Rewriting query with {len(url_changes)} URL change(s)...")

            rewritten_query = self._rewrite_query(query, url_changes, language)

            return AgentOutput(
                success=True,
                data={"rewritten_query": rewritten_query, "query_was_rewritten": True},
            )

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return AgentOutput(
                success=False,
                errors=[f"Query rewrite error: {str(e)}"],
                data={"rewritten_query": query, "query_was_rewritten": False},
            )

    def _rewrite_query(self, query: str, url_changes: List[Dict], language: str) -> str:
        """Rewrite query with URL changes using LLM"""

        prompt = self._build_rewrite_prompt(query, url_changes, language)

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.llm_client.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
            )

            content = response.choices[0].message.content

            # Parse JSON response
            response_text = content.strip()
            if "```json" in response_text:
                response_text = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text)
            rewritten_query = data.get("rewritten_query", query)

            logger.info("Query rewritten successfully")
            return rewritten_query

        except Exception as e:
            logger.error(f"Failed to rewrite query: {e}")
            # Fallback: simple string replacement
            rewritten = query
            for change in url_changes:
                if change["action"] == "replace":
                    rewritten = rewritten.replace(
                        change["original_url"], change["new_url"]
                    )
                elif change["action"] == "remove":
                    rewritten = rewritten.replace(
                        change["original_url"], "[URL removed]"
                    )

            return rewritten

    def _build_rewrite_prompt(
        self, query: str, url_changes: List[Dict], language: str
    ) -> str:
        """Build the LLM prompt for query rewriting"""

        if language.lower() == "chinese":
            changes_text = ""
            for change in url_changes:
                if change["action"] == "replace":
                    changes_text += (
                        f"\n- 替换: {change['original_url']} → {change['new_url']}"
                    )
                elif change["action"] == "remove":
                    changes_text += f"\n- 移除: {change['original_url']}"

            return f"""你是查询重写专家。根据以下URL变更重写查询。

原始查询：
{query}

URL变更：{changes_text}

要求：
1. 保持查询的核心意图和任务不变
2. 如果URL被替换：用新URL替换旧URL，确保自然流畅
3. 如果URL被移除：移除该URL，并建议用户寻找替代数据源或自行搜索该数据集
4. 重写后的查询应该读起来自然，就像用户自己写的

返回JSON格式：
{{
  "rewritten_query": "重写后的查询",
  "changes_made": ["变更说明1", "变更说明2"]
}}

请直接返回JSON，不要额外说明。"""
        else:
            changes_text = ""
            for change in url_changes:
                if change["action"] == "replace":
                    changes_text += (
                        f"\n- Replace: {change['original_url']} → {change['new_url']}"
                    )
                elif change["action"] == "remove":
                    changes_text += f"\n- Remove: {change['original_url']}"

            return f"""You are a query rewriting expert. Rewrite the query based on the following URL changes.

Original query:
{query}

URL changes:{changes_text}

Requirements:
1. Keep the core intent and task of the query unchanged
2. If URL is replaced: replace the old URL with the new one, ensuring natural flow
3. If URL is removed: remove the URL and suggest the user find alternative data sources or search for the dataset themselves
4. The rewritten query should read naturally, as if written by the user

Return JSON format:
{{
  "rewritten_query": "rewritten query",
  "changes_made": ["change description 1", "change description 2"]
}}

Return ONLY JSON, no extra explanation."""
