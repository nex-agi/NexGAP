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
Agent that determines whether a generated query requires provided files.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from .base import Agent, AgentContext, AgentOutput

if TYPE_CHECKING:
    from llm_interface.query_generator import LLMClient, PromptTemplateManager

logger = logging.getLogger(__name__)


class FileRequirementAgent(Agent):
    """Decides if a query explicitly depends on files supplied by the system."""

    def __init__(
        self, llm_client: "LLMClient", prompt_manager: "PromptTemplateManager"
    ):
        super().__init__("file_requirement")
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    def run(self, context: AgentContext) -> AgentOutput:
        query_text: Optional[str] = context.get("current_query_text")
        language: str = (context.get("language") or "english").lower()

        if not query_text:
            logger.debug("FileRequirementAgent skipped: no query text available")
            return AgentOutput(success=True)

        prompt = self.prompt_manager.format_file_requirement_prompt(
            query=query_text, language=language
        )

        start_time = time.time()
        try:
            response = self.llm_client.generate_completion(prompt)
            elapsed = time.time() - start_time

            parsed = self._parse_response(response)
            requires_files = parsed.get("requires_files", False)
            reason = parsed.get("reason", "").strip()
            required_items = parsed.get("required_items", [])

            context.set("query_requires_files", requires_files)
            context.set("query_file_requirement_reason", reason)
            context.set("query_required_items", required_items)

            if requires_files:
                logger.info(
                    "Query marked as file-dependent: %s",
                    reason or "reason not provided",
                )

            return AgentOutput(
                success=True,
                data={
                    "query_requires_files": requires_files,
                    "query_file_requirement_reason": reason,
                    "query_required_items": required_items,
                },
                timings={"analysis": elapsed},
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("File requirement analysis failed: %s", exc)
            context.append_error(f"file_requirement: {exc}")
            return AgentOutput(success=False, errors=[str(exc)])

    def _parse_response(self, response: str) -> Dict[str, any]:
        """Parse JSON response from LLM safely."""
        try:
            json_str = self._extract_json(response)
            if not json_str:
                return {}
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse file requirement JSON: %s; response=%s", exc, response
            )
            data = {}

        requires_files = bool(data.get("requires_files"))
        reason = data.get("reason", "")
        required_items = data.get("required_items") or []
        if not isinstance(required_items, list):
            required_items = []

        return {
            "requires_files": requires_files,
            "reason": reason,
            "required_items": [str(item) for item in required_items if item],
        }

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the first JSON object from the text."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]
