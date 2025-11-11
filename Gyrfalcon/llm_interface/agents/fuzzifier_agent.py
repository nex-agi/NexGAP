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
Agent responsible for softening synthesized queries into human-like, implicit requests.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from .base import Agent, AgentContext, AgentOutput

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llm_interface.query_generator import (
        GeneratedQuery,
        LLMClient,
        PromptTemplateManager,
    )

logger = logging.getLogger(__name__)
FAILURE_LOG = Path("logs/fuzzifier_failures.jsonl")


@dataclass
class FuzzifierResult:
    """Structured result from fuzzifying a query."""

    analysis: str
    fuzzy_query: str
    strategy: Optional[str] = None


class FuzzifierAgent(Agent):
    """Applies probabilistic fuzzification to generated queries."""

    def __init__(
        self,
        llm_client: "LLMClient",
        prompt_manager: "PromptTemplateManager",
        probability: float = 0.0,
    ) -> None:
        super().__init__("query_fuzzifier")
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.probability = self._clamp_probability(probability)
        self.failure_log = FAILURE_LOG

    @staticmethod
    def _clamp_probability(probability: float) -> float:
        return max(0.0, min(1.0, float(probability or 0.0)))

    def set_probability(self, probability: float) -> None:
        """Update invocation probability."""
        self.probability = self._clamp_probability(probability)

    def run(self, context: AgentContext) -> AgentOutput:
        """Conditionally fuzzify the current query within the agent context."""
        if self.probability <= 0.0:
            logger.debug("🚫 FuzzifierAgent disabled (probability: 0.0)")
            return AgentOutput(success=True)

        query_obj: Optional["GeneratedQuery"] = context.get("current_query_object")
        query_text: Optional[str] = None

        if query_obj:
            query_text = getattr(query_obj, "content", None)

        if not query_text:
            query_text = context.get("current_query_text")

        if not query_text:
            logger.debug("⏭️ FuzzifierAgent skipped: no query text available")
            return AgentOutput(success=True)

        if random.random() >= self.probability:
            logger.debug(
                "🎲 FuzzifierAgent skipped due to probability gate (p=%.2f)",
                self.probability,
            )
            return AgentOutput(success=True)

        logger.info(
            f"🔮 FuzzifierAgent processing query (probability: {self.probability:.2f})"
        )
        logger.debug(
            f"📝 Original query: {query_text[:100]}{'...' if len(query_text) > 100 else ''}"
        )

        try:
            result, elapsed = self.fuzzify_text(query_text)
            logger.info(f"✅ Fuzzification completed in {elapsed:.2f}s")
            logger.debug(
                f"🎭 Fuzzified query: {result.fuzzy_query[:100]}{'...' if len(result.fuzzy_query) > 100 else ''}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("❌ FuzzifierAgent failed: %s", exc)
            self._record_error_metadata(query_obj, str(exc))
            return AgentOutput(success=True, errors=[str(exc)])

        self._apply_fuzzified_result(query_obj, query_text, result)

        if query_obj:
            context.set("current_query_object", query_obj)
            context.set("current_query_text", query_obj.content)
        else:
            context.set("current_query_text", result.fuzzy_query)

        return AgentOutput(
            success=True,
            data={
                "fuzzifier_analysis": result.analysis,
                "fuzzifier_strategy": result.strategy,
            },
            timings={"fuzzify": elapsed},
        )

    # Public helper -----------------------------------------------------

    def fuzzify_text(self, query: str) -> tuple[FuzzifierResult, float]:
        """Fuzzify a query and return the result along with elapsed time."""
        prompt = self.prompt_manager.format_fuzzifier_prompt(query)
        start_time = time.time()
        response = self.llm_client.generate_completion(prompt).strip()
        elapsed = time.time() - start_time
        result = self._parse_response(query, response)
        return result, elapsed

    # Internal helpers --------------------------------------------------

    def _apply_fuzzified_result(
        self,
        query_obj: Optional["GeneratedQuery"],
        original_text: str,
        result: FuzzifierResult,
    ) -> None:
        """Mutate the query object and metadata with fuzzifier output."""
        metadata: Dict[str, object]
        if query_obj:
            metadata = query_obj.metadata or {}
        else:
            metadata = {}

        fuzz_meta = metadata.get("fuzzifier", {})
        fuzz_meta.update(
            {
                "attempted": True,
                "applied": True,
                "probability": self.probability,
                "analysis": result.analysis,
                "original_query": original_text,
            }
        )
        if result.strategy:
            fuzz_meta["strategy"] = result.strategy
        else:
            fuzz_meta.pop("strategy", None)
        fuzz_meta.pop("error", None)

        metadata["fuzzifier"] = fuzz_meta

        if query_obj:
            query_obj.content = result.fuzzy_query
            query_obj.metadata = metadata

    def _record_error_metadata(
        self,
        query_obj: Optional["GeneratedQuery"],
        error_message: str,
    ) -> None:
        """Record failure information in metadata and failure log."""
        if query_obj:
            metadata = query_obj.metadata or {}
        else:
            metadata = {}

        fuzz_meta = metadata.get("fuzzifier", {})
        fuzz_meta.update(
            {
                "attempted": True,
                "applied": False,
                "probability": self.probability,
                "error": error_message,
            }
        )
        fuzz_meta.pop("analysis", None)
        fuzz_meta.pop("strategy", None)
        metadata["fuzzifier"] = fuzz_meta

        if query_obj:
            query_obj.metadata = metadata

    def _parse_response(self, query: str, response: str) -> FuzzifierResult:
        """Parse JSON payload from the LLM response."""
        try:
            json_payload = self._extract_json_object(response)
            data = json.loads(json_payload)
        except Exception as exc:  # noqa: BLE001
            self._log_failure(query, response)
            raise ValueError(f"Failed to parse fuzzifier response: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Fuzzifier response is not a JSON object: {data!r}")

        analysis = str(data.get("analysis", "")).strip()
        fuzzy_query = str(data.get("fuzzy_query", "")).strip()
        strategy = str(data.get("strategy", "")).strip() or None

        if not fuzzy_query:
            self._log_failure(query, response)
            raise ValueError("Fuzzifier response lacks fuzzy_query")

        return FuzzifierResult(
            analysis=analysis, fuzzy_query=fuzzy_query, strategy=strategy
        )

    def _log_failure(self, query: str, response: str) -> None:
        """Persist malformed responses for offline inspection."""
        try:
            self.failure_log.parent.mkdir(parents=True, exist_ok=True)
            with self.failure_log.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"query": query, "response": response}, ensure_ascii=False
                    )
                    + "\n"
                )
        except Exception:  # pragma: no cover - best effort logging
            logger.debug("Failed to write fuzzifier failure log", exc_info=True)

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Return the JSON object substring from a chunk of text."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object detected in fuzzifier response")
        return text[start : end + 1]
