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
URL Extraction Agent for Gyrfalcon v5

Extracts valid URLs from query text using LLM, filtering out placeholders.
"""

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from .base import Agent, AgentContext, AgentOutput

logger = logging.getLogger(__name__)


class URLExtractionAgent(Agent):
    """
    Extracts valid URLs from query text using LLM analysis.
    Filters out placeholder URLs like {variable}, <placeholder>, example.com
    """

    def __init__(self, llm_client, name: str = "URLExtractionAgent"):
        super().__init__(name)
        self.llm_client = llm_client

    def run(self, context: AgentContext) -> AgentOutput:
        """
        Extract URLs from the query.

        Expected context keys:
            - query: str (the query text to extract URLs from)
            - language: str (optional, "english" or "chinese")

        Adds to context:
            - extracted_urls: List[Dict] (list of extracted URL info)
            - has_urls: bool (whether URLs were found)
        """
        query = context.get("query")
        language = context.get("language", "english")

        if not query:
            return AgentOutput(
                success=False, errors=["No query provided for URL extraction"]
            )

        try:
            logger.info(f"Extracting URLs from query: {query[:100]}...")

            extracted_urls = self._extract_urls(query, language)

            return AgentOutput(
                success=True,
                data={
                    "extracted_urls": extracted_urls,
                    "has_urls": len(extracted_urls) > 0,
                },
            )

        except Exception as e:
            logger.error(f"URL extraction failed: {e}")
            return AgentOutput(
                success=False, errors=[f"URL extraction error: {str(e)}"]
            )

    def _extract_urls(self, query: str, language: str) -> List[Dict[str, Any]]:
        """Use LLM to extract valid URLs from query text"""

        if language.lower() == "chinese":
            prompt = f"""你是URL提取专家。从以下查询中提取所有真实的、可直接访问的URL。

查询：
{query}

要求：
1. 只提取真实的URL（必须是http://或https://开头）
2. 排除占位符URL（如包含{{}}、<>等）
3. 对每个URL，提供简短描述和在查询中的用途

返回JSON格式：
{{
  "urls": [
    {{
      "url": "完整URL",
      "description": "URL内容描述",
      "context": "在查询中的用途"
    }}
  ]
}}

如果没有找到真实URL，返回空列表。请直接返回JSON，不要额外说明。"""
        else:
            prompt = f"""You are a URL extraction expert. Extract all real, directly accessible URLs from the following query.

Query:
{query}

Requirements:
1. Only extract real URLs (must start with http:// or https://)
2. Exclude placeholder URLs (containing {{}}, <>, etc.)
3. For each URL, provide a brief description and its purpose in the query

Return JSON format:
{{
  "urls": [
    {{
      "url": "complete URL",
      "description": "URL content description",
      "context": "purpose in the query"
    }}
  ]
}}

If no real URLs are found, return an empty list. Return ONLY JSON, no extra explanation."""

        try:
            response = self.llm_client.generate_completion(prompt)

            # Log the raw response for debugging
            logger.debug(
                f"Raw LLM response ({len(response)} chars): {repr(response[:500])}"
            )

            # Parse JSON response
            # Try to extract JSON from response
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            logger.debug(f"Cleaned response for parsing: {repr(response[:300])}")

            data = json.loads(response)
            urls = data.get("urls", [])

            # Add is_placeholder flag
            for url_info in urls:
                url_info["is_placeholder"] = False

            logger.info(f"Extracted {len(urls)} URL(s)")
            return urls

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Full response text: {response}")
            logger.error(f"Response length: {len(response)} characters")
            return []
        except Exception as e:
            logger.error(f"URL extraction error: {e}")
            return []
