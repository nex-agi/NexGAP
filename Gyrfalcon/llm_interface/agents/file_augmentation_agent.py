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
Agent that rewrites queries to incorporate provided local file paths naturally.
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


class FileAugmentationAgent(Agent):
    """Produces an updated query that references downloaded files naturally."""

    def __init__(
        self, llm_client: "LLMClient", prompt_manager: "PromptTemplateManager"
    ):
        super().__init__("file_augmentation")
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    def run(self, context: AgentContext) -> AgentOutput:
        query_text: Optional[str] = context.get("current_query_text")
        downloaded_files: List[Dict[str, str]] = context.get("downloaded_files", [])
        language: str = (context.get("language") or "english").lower()

        usable_files = [
            file_info
            for file_info in downloaded_files
            if file_info.get("status") == "downloaded"
        ]

        if not query_text or not usable_files:
            logger.debug("FileAugmentationAgent skipped: no files to reference.")
            return AgentOutput(success=True)

        prompt = self.prompt_manager.format_file_query_rewrite_prompt(
            query=query_text, files=usable_files, language=language
        )

        rewrite_start = time.time()
        try:
            response = self.llm_client.generate_completion(prompt)
            rewrite_elapsed = time.time() - rewrite_start

            updated_query = self._extract_rewritten_query(response)
            if not updated_query:
                logger.warning(
                    "Failed to extract rewritten query; keeping original text."
                )
                updated_query = query_text

            context.set("augmented_query_text", updated_query)
            context.set("usable_downloaded_files", usable_files)

            metadata_update = {"file_system": {"files": usable_files}}

            return AgentOutput(
                success=True,
                data={
                    "augmented_query_text": updated_query,
                    "file_metadata_update": metadata_update,
                },
                timings={"rewrite": rewrite_elapsed},
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("File augmentation failed: %s", exc)
            context.append_error(f"file_augmentation: {exc}")
            return AgentOutput(success=False, errors=[str(exc)])

    @staticmethod
    def _extract_rewritten_query(response: str) -> Optional[str]:
        response = response.strip()
        if not response:
            return None

        # Prefer JSON output with "rewritten_query"
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(response[start : end + 1])
                rewritten = data.get("rewritten_query")
                if rewritten:
                    return rewritten.strip()
            except json.JSONDecodeError:
                pass

        return response
