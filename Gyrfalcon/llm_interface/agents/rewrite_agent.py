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
RewriteAgent adapts personas to fit selected problem types before query synthesis.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from frameworks.framework_manager import FrameworkPersona

from .base import Agent, AgentContext, AgentOutput

if TYPE_CHECKING:
    from llm_interface.query_generator import LLMClient, PromptTemplateManager

logger = logging.getLogger(__name__)


class RewriteAgent(Agent):
    """
    Agent responsible for evaluating and rewriting personas when necessary.
    """

    def __init__(
        self, llm_client: "LLMClient", prompt_manager: "PromptTemplateManager"
    ):
        super().__init__(name="rewrite")
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager

    def run(self, context: AgentContext) -> AgentOutput:
        persona = context.get("persona")
        problem_type = context.get("problem_type")
        language = (context.get("language") or "english").lower()
        framework_name = context.get("framework_name")

        if not isinstance(persona, FrameworkPersona):
            error = "RewriteAgent requires 'persona' in context as FrameworkPersona"
            logger.error(error)
            return AgentOutput(success=False, errors=[error])

        if not problem_type:
            error = "RewriteAgent requires 'problem_type' in context"
            logger.error(error)
            return AgentOutput(success=False, errors=[error])

        timings: Dict[str, float] = {}
        run_start = time.time()

        persona_text = persona.get_persona(language)
        persona_text_english = persona.persona
        persona_text_chinese = persona.persona_chinese

        self._print_header(persona_text, problem_type, language)

        evaluation_start = time.time()
        is_suitable = self._evaluate_persona_suitability(
            persona_text=persona_text, problem_type=problem_type, language=language
        )
        timings["persona_evaluation"] = time.time() - evaluation_start

        if is_suitable:
            self._print_suitability(language)
            timings["persona_adaptation"] = time.time() - run_start
            return AgentOutput(
                success=True,
                data={
                    "persona": persona,
                    "persona_was_rewritten": False,
                },
                timings=timings,
            )

        self._print_rewrite_notice(language)

        rewrite_start = time.time()
        rewritten_text_english = self._rewrite_persona(
            persona_text=persona_text_english or persona_text,
            problem_type=problem_type,
            language="english",
        )

        rewritten_text_chinese = self._rewrite_persona(
            persona_text=persona_text_chinese or persona_text,
            problem_type=problem_type,
            language="chinese",
        )
        timings["persona_rewriting"] = time.time() - rewrite_start

        adapted_persona = FrameworkPersona(
            persona=rewritten_text_english, persona_chinese=rewritten_text_chinese
        )

        save_time = self._append_persona_to_file(
            rewritten_text_english, rewritten_text_chinese, framework_name
        )
        if save_time is not None:
            timings["persona_save"] = save_time

        self._print_rewrite_result(
            language, rewritten_text_english, rewritten_text_chinese
        )

        timings["persona_adaptation"] = time.time() - run_start

        return AgentOutput(
            success=True,
            data={
                "persona": adapted_persona,
                "persona_was_rewritten": True,
                "rewritten_persona": adapted_persona,
            },
            timings=timings,
        )

    def _evaluate_persona_suitability(
        self, persona_text: str, problem_type: str, language: str
    ) -> bool:
        """Evaluate if the persona fits the problem type."""
        prompt = self.prompt_manager.format_persona_evaluation_prompt(
            persona=persona_text, problem_type=problem_type, language=language
        )

        response = self.llm_client.generate_completion(prompt)
        result = response.strip().upper()
        is_suitable = "SUITABLE" in result and "NOT" not in result

        logger.debug(
            "Persona evaluation response: '%s' (suitable=%s)", result, is_suitable
        )
        return is_suitable

    def _rewrite_persona(
        self, persona_text: str, problem_type: str, language: str
    ) -> str:
        """Rewrite persona using LLM if unsuitable."""
        prompt = self.prompt_manager.format_persona_rewriting_prompt(
            persona=persona_text, problem_type=problem_type, language=language
        )

        response = self.llm_client.generate_completion(prompt)
        rewritten = response.strip()

        if not rewritten:
            logger.warning("LLM returned empty rewrite; using original persona")
            return persona_text

        return rewritten

    def _append_persona_to_file(
        self,
        persona_text: str,
        persona_text_chinese: str,
        framework_name: Optional[str],
    ) -> Optional[float]:
        """Persist rewritten persona to the framework persona file."""
        if not framework_name:
            return None

        start_time = time.time()

        try:
            frameworks_dir = Path(__file__).parent.parent.parent / "frameworks"
            persona_file = frameworks_dir / framework_name / "persona.jsonl"

            if not persona_file.exists():
                logger.warning("Persona file not found: %s", persona_file)
                return time.time() - start_time

            persona_entry = {
                "persona": persona_text,
                "persona_chinese": persona_text_chinese,
            }

            with open(persona_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(persona_entry, ensure_ascii=False) + "\n")

            logger.info("Appended rewritten persona to %s", persona_file)

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to append persona rewrite: %s", exc)

        return time.time() - start_time

    @staticmethod
    def _print_header(persona_text: str, problem_type: str, language: str) -> None:
        print("\n" + "=" * 80)
        print("📋 PERSONA ADAPTATION CHECK")
        print("=" * 80)
        if language == "chinese":
            print(f"原始角色: {persona_text}")
            print(f"问题类型: {problem_type}")
            print("语言: 中文")
        else:
            print(f"Original Persona: {persona_text}")
            print(f"Problem Type: {problem_type}")
            print("Language: english")
        print("-" * 80)

    @staticmethod
    def _print_suitability(language: str) -> None:
        if language == "chinese":
            print("✅ 该角色适合此问题类型 - 无需重写")
        else:
            print("✅ Persona is SUITABLE for this problem type - no rewriting needed")
        print("=" * 80 + "\n")

    @staticmethod
    def _print_rewrite_notice(language: str) -> None:
        if language == "chinese":
            print("❌ 该角色不适合此问题类型 - 正在重写...")
        else:
            print("❌ Persona is NOT SUITABLE for this problem type - rewriting...")

    @staticmethod
    def _print_rewrite_result(language: str, english: str, chinese: str) -> None:
        if language == "chinese":
            print(f"📝 重写后的角色: {chinese}")
            print("✅ 角色已成功适配并保存")
        else:
            print(f"📝 Rewritten Persona: {english}")
            print("✅ Persona successfully adapted and saved")
        print("=" * 80 + "\n")
