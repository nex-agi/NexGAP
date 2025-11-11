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
Agent that optionally enriches the query context with web search results.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from textwrap import shorten
from typing import Dict, List, Optional

import requests

from .base import Agent, AgentContext, AgentOutput

logger = logging.getLogger(__name__)

WEB_TAG = "🌐 >>>WEB<<<"


@dataclass
class WebSearchConfig:
    """Configuration for the web research agent."""

    probability: float = 0.0
    api_key: Optional[str] = None
    max_results: int = 5
    endpoint: str = "https://google.serper.dev/search"
    timeout: float = 8.0
    market: str = "us"
    language: str = "en"
    enable: bool = field(init=False, default=False)

    def __post_init__(self):
        self.probability = max(0.0, min(1.0, float(self.probability)))
        self.max_results = max(1, int(self.max_results or 1))
        self.enable = self.probability > 0 and bool(self.api_key)


class WebResearchAgent(Agent):
    """
    Agent that performs optional web research to provide fresh context.
    Uses the Serper.dev API by default.
    """

    def __init__(self, config: Optional[Dict[str, object]] = None):
        super().__init__("web_research")
        self.config = WebSearchConfig(**(config or {}))

    def update_config(self, config: Optional[Dict[str, object]]) -> None:
        """Update runtime configuration."""
        self.config = WebSearchConfig(**(config or {}))

    def run(self, context: AgentContext) -> AgentOutput:
        cfg = self.config
        if not cfg.enable:
            context.set("search_context", None)
            logger.debug(
                "%s Web search disabled (probability=0 or missing API key)", WEB_TAG
            )
            return AgentOutput(success=True)

        roll = random.random()
        if roll >= cfg.probability:
            logger.debug(
                "%s Skipped (roll %.3f >= prob %.3f)", WEB_TAG, roll, cfg.probability
            )
            context.set("search_context", None)
            return AgentOutput(success=True)

        persona_obj = context.get("persona")
        problem_type = context.get("problem_type")
        language = (context.get("language") or "english").lower()

        if not persona_obj or not problem_type:
            context.set("search_context", None)
            return AgentOutput(success=True)

        persona_text = getattr(persona_obj, "get_persona", lambda _: str(persona_obj))(
            language
        )
        queries = self._build_queries(persona_text, problem_type)

        start_time = time.time()
        aggregated: List[Dict[str, object]] = []
        errors: List[str] = []

        for query in queries:
            try:
                results = self._run_serper_search(query, cfg)
                for item in results:
                    item["search_query"] = query
                aggregated.extend(results)
            except requests.RequestException as exc:
                errors.append(str(exc))
                logger.warning(
                    "%s Request failed for query '%s': %s", WEB_TAG, query, exc
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                logger.exception(
                    "%s Unexpected error during web search for '%s': %s",
                    WEB_TAG,
                    query,
                    exc,
                )

        search_context = {
            "queries": queries,
            "results": aggregated[: cfg.max_results],
            "provider": "serper",
            "duration": time.time() - start_time,
            "errors": errors,
            "timestamp": time.time(),
            "used": True,
        }
        context.set("search_context", search_context)

        logger.info(
            "%s Executed (%d queries, %d results kept, errors=%d)",
            WEB_TAG,
            len(queries),
            len(search_context["results"]),
            len(errors),
        )

        return AgentOutput(success=True, data={"search_context": search_context})

    @staticmethod
    def _build_queries(persona: str, problem_type: str) -> List[str]:
        persona_fragment = shorten(persona, width=80, placeholder="")
        candidates = [
            problem_type.strip(),
            f"{persona_fragment} {problem_type}".strip(),
        ]
        seen = set()
        deduped = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped or [problem_type]

    @staticmethod
    def _run_serper_search(query: str, cfg: WebSearchConfig) -> List[Dict[str, object]]:
        headers = {"X-API-KEY": cfg.api_key, "Content-Type": "application/json"}
        payload = {
            "q": query,
            "num": cfg.max_results,
            "gl": cfg.market,
            "hl": cfg.language,
        }

        response = requests.post(
            cfg.endpoint, headers=headers, json=payload, timeout=cfg.timeout
        )
        response.raise_for_status()
        data = response.json()

        organic = data.get("organic") or []
        results: List[Dict[str, object]] = []
        for item in organic[: cfg.max_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "url": item.get("link"),
                    "source": item.get("source"),
                    "date": item.get("date"),
                }
            )

        # Include top stories if organic is empty
        if not results:
            stories = data.get("news") or []
            for story in stories[: cfg.max_results]:
                results.append(
                    {
                        "title": story.get("title"),
                        "snippet": story.get("snippet"),
                        "url": story.get("link"),
                        "source": story.get("source"),
                        "date": story.get("date"),
                    }
                )
        return results
