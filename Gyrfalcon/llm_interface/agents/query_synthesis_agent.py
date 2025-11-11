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
QuerySynthesisAgent coordinates prompt construction, LLM calls, and parsing.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import Agent, AgentContext, AgentOutput

if TYPE_CHECKING:
    from frameworks.framework_manager import FrameworkPersona
    from llm_interface.query_generator import (
        LLMClient,
        PromptTemplateManager,
        ResponseParser,
    )
    from problem_type_tree import TagTrace

logger = logging.getLogger(__name__)


class QuerySynthesisAgent(Agent):
    """
    Agent responsible for generating query prompts, invoking the LLM,
    parsing responses, and selecting final queries.
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        prompt_manager: "PromptTemplateManager",
        response_parser: "ResponseParser",
        generated_query_cls,
        result_cls,
    ):
        super().__init__(name="query_synthesis")
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.response_parser = response_parser
        self.generated_query_cls = generated_query_cls
        self.result_cls = result_cls

    def run(self, context: AgentContext) -> AgentOutput:
        persona = context.get("persona")
        problem_type = context.get("problem_type")
        tag_trace = context.get("tag_trace")
        language = (context.get("language") or "english").lower()
        debug_mode = bool(context.get("debug_mode"))
        difficulty_distribution = context.get("difficulty_distribution")
        framework_name = context.get("framework_name")
        search_context = context.get("search_context")
        framework_description = context.get("framework_description")

        if persona is None:
            error = "QuerySynthesisAgent requires 'persona' in context"
            logger.error(error)
            return AgentOutput(success=False, errors=[error])

        if not problem_type:
            error = "QuerySynthesisAgent requires 'problem_type' in context"
            logger.error(error)
            return AgentOutput(success=False, errors=[error])

        if tag_trace is None or not hasattr(tag_trace, "get_labels"):
            error = "QuerySynthesisAgent requires 'tag_trace' supporting get_labels()"
            logger.error(error)
            return AgentOutput(success=False, errors=[error])

        timings: Dict[str, float] = {}
        run_start = time.time()

        prompt_start = time.time()
        prompt = self.prompt_manager.format_query_generation_prompt(
            persona=persona,
            tag_trace=tag_trace,
            problem_type=problem_type,
            language=language,
            search_context=search_context,
            framework_description=framework_description,
        )
        timings["prompt_formatting"] = time.time() - prompt_start

        if debug_mode:
            result = self._save_prompt_and_stop(
                prompt=prompt,
                persona=persona,
                tag_trace=tag_trace,
                problem_type=problem_type,
                language=language,
                search_context=search_context,
            )
            timings["query_generation_total"] = time.time() - run_start
            # Timing information for debug mode is attached by QueryGenerator.
            return AgentOutput(
                success=True, data={"query_result": result}, timings=timings
            )

        response_start = time.time()
        response = self.llm_client.generate_completion(prompt)
        timings["llm_api_call"] = time.time() - response_start

        parsing_start = time.time()
        queries = self.response_parser.parse_query_generation_response(
            response=response, language=language
        )
        timings["response_parsing"] = time.time() - parsing_start

        selection_start = time.time()
        selected_query = self._select_query_by_difficulty(
            queries=queries, difficulty_distribution=difficulty_distribution
        )
        timings["difficulty_selection"] = time.time() - selection_start

        timings["query_generation_total"] = time.time() - run_start

        result = self.result_cls(
            queries=[selected_query] if selected_query else [],
            trace_context=tag_trace.get_labels(),
            problem_type=problem_type,
            generation_metadata={
                "persona": getattr(persona, "persona", ""),
                "problem_type": problem_type,
                "language": language,
                "timestamp": time.time(),
                "raw_response": response,
                "framework": framework_name,
                "all_queries_count": len(queries),
                "selected_difficulty": getattr(selected_query, "difficulty", None),
                "search_context": search_context,
            },
        )

        return AgentOutput(success=True, data={"query_result": result}, timings=timings)

    def _select_query_by_difficulty(
        self, queries: List, difficulty_distribution: Optional[Dict[str, float]] = None
    ):
        """Randomly select a query based on the configured difficulty distribution."""
        if not queries:
            return None

        difficulty_map = {}
        for query in queries:
            difficulty = getattr(query, "difficulty", "").lower()
            if difficulty:
                difficulty_map[difficulty] = query

        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.20, "medium": 0.50, "hard": 0.30}

        difficulties = list(difficulty_distribution.keys())
        probabilities = list(difficulty_distribution.values())

        selected_difficulty = random.choices(difficulties, weights=probabilities, k=1)[
            0
        ]
        selected_query = difficulty_map.get(selected_difficulty)

        if selected_query:
            logger.info("Selected query difficulty: %s", selected_difficulty)
            return selected_query

        logger.warning(
            "Selected difficulty '%s' not found, using first query", selected_difficulty
        )
        return queries[0]

    def _save_prompt_and_stop(
        self,
        prompt: str,
        persona: "FrameworkPersona",
        tag_trace: "TagTrace",
        problem_type: str,
        language: str,
        search_context: Optional[Dict[str, Any]] = None,
    ):
        """Persist prompt to file and return a debug-mode result."""
        timestamp = int(time.time())
        debug_dir = Path("debug_prompts")
        debug_dir.mkdir(exist_ok=True)

        persona_short = persona.persona[:30].replace(" ", "_").replace("/", "_")
        problem_type_short = problem_type[:20].replace(" ", "_").replace("/", "_")
        debug_file = (
            debug_dir / f"prompt_{persona_short}_{problem_type_short}_{timestamp}.txt"
        )

        search_context_json = (
            json.dumps(search_context, ensure_ascii=False, indent=2)
            if search_context
            else "{}"
        )

        debug_content = f"""=== DEBUG PROMPT SAVE ===
Timestamp: {timestamp}
Problem Type: {problem_type}
Language: {language}

=== SEARCH CONTEXT ===
{search_context_json}

=== PROMPT ===
{prompt}

=== END OF PROMPT ===
"""
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(debug_content)

        logger.info("Debug prompt saved to %s", debug_file)

        return self.result_cls(
            queries=[],
            trace_context=tag_trace.get_labels(),
            problem_type=problem_type,
            generation_metadata={
                "debug_mode": True,
                "prompt_file": str(debug_file),
                "persona": persona.persona,
                "problem_type": problem_type,
                "language": language,
                "timestamp": timestamp,
                "debug_message": "Debug mode: prompt saved, LLM call skipped",
            },
        )
