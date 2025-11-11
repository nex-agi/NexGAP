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
LLM Interface and Query Generation System for Gyrfalcon v3 Pipeline

This module handles communication with LLM services and generates queries
based on problem type traces, personas, and framework metadata.
"""

import copy
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai

from frameworks.framework_manager import FrameworkPersona
from llm_interface.agents import (
    AgentContext,
    FileAugmentationAgent,
    FileRequirementAgent,
    FileSystemAgent,
    QuerySynthesisAgent,
    RewriteAgent,
    RouterAgent,
    URLProcessingAgent,
)
from llm_interface.agents.fuzzifier_agent import FuzzifierAgent
from llm_interface.agents.web_research_agent import WebResearchAgent
from problem_type_tree import ProblemTypeNode, TagTrace

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuery:
    """Represents a generated query with metadata"""

    content: str
    difficulty: str
    context: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class QueryGenerationResult:
    """Result of query generation including queries (no new tags in new design)"""

    queries: List[GeneratedQuery]
    trace_context: List[str]
    problem_type: str  # The selected problem type
    generation_metadata: Dict[str, Any]


class LLMClient:
    """
    Client for communicating with LLM services using OpenAI API.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 600.0,
        max_retries: int = 3,
    ):
        self.client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_completion(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a completion from the LLM.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content

            # Debug: Check if content is None or empty
            if not content:
                logger.error(f"LLM returned empty content! Response object: {response}")
                logger.error(f"Choices: {response.choices}")
                logger.error(
                    f"Message: {response.choices[0].message if response.choices else 'No choices'}"
                )
                return ""

            logger.debug(f"LLM response length: {len(content)} characters")
            return content

        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise


class PromptTemplateManager:
    """
    Manages prompt templates for query generation.
    """

    def __init__(self):
        self.templates = {
            "persona_evaluation_english": """You are an expert at evaluating whether a user persona is suitable for asking questions about a specific problem type.

Given:
- User Persona: {persona}
- Problem Type: "{problem_type}"

Task: Evaluate if this persona would naturally ask questions about this problem type.

Respond ONLY with one of these exact phrases:
- "SUITABLE" - if the persona would naturally ask about this problem type
- "NOT_SUITABLE" - if the persona would NOT naturally ask about this problem type

Consider:
1. Does the persona's background/role/expertise align with this problem type?
2. Would someone with this persona have legitimate reasons to ask about this problem type?
3. Is there a natural fit between the persona's needs and this problem type?

Your response:""",
            "persona_evaluation_chinese": """你是评估用户角色是否适合询问特定问题类型的专家。

给定信息：
- 用户角色: {persona}
- 问题类型: "{problem_type}"

任务：评估这个用户角色是否会自然地询问关于这个问题类型的问题。

请只回答以下短语之一：
- "SUITABLE" - 如果该角色会自然地询问这个问题类型
- "NOT_SUITABLE" - 如果该角色不会自然地询问这个问题类型

考虑因素：
1. 角色的背景/职位/专业知识是否与此问题类型一致？
2. 具有此角色的人是否有合理的理由询问此问题类型？
3. 角色的需求与此问题类型之间是否有自然的契合？

你的回答：""",
            "persona_rewriting_english_simple": """You are an expert at adapting user personas to make them suitable for asking questions about specific problem types.

Given:
- Original Persona: {persona}
- Problem Type: "{problem_type}"

Task: Rewrite the persona in THIRD PERSON (describing "a person who...") to make it suitable for asking about this problem type. Keep it CONCISE and natural.

Requirements:
1. Use THIRD PERSON description (e.g., "A researcher who..." NOT "I am...")
2. Keep it SHORT and focused (1-2 sentences maximum)
3. Preserve key attributes (role, constraints)
4. Adjust context to fit the problem type
5. Write in the same language as the original

Output ONLY the rewritten persona, nothing else.

Rewritten persona:""",
            "persona_rewriting_english_moderate": """You are an expert at adapting user personas to make them suitable for asking questions about specific problem types.

Given:
- Original Persona: {persona}
- Problem Type: "{problem_type}"

Task: Rewrite the persona in THIRD PERSON (describing "a person who...") to make it suitable for asking about this problem type. Use moderate detail.

Requirements:
1. Use THIRD PERSON description (e.g., "A professor who..." NOT "I...")
2. Keep it CONCISE (2-3 sentences)
3. Preserve core characteristics (career stage, resources, work context)
4. Adjust situation to align with the problem type
5. Write naturally in the same language as the original

Output ONLY the rewritten persona, nothing else.

Rewritten persona:""",
            "persona_rewriting_english_detailed": """You are an expert at adapting user personas to make them suitable for asking questions about specific problem types.

Given:
- Original Persona: {persona}
- Problem Type: "{problem_type}"

Task: Rewrite the persona in THIRD PERSON (describing "a person who...") to make it suitable for asking about this problem type. Include relevant details.

Requirements:
1. Use THIRD PERSON description (e.g., "A senior researcher who..." NOT "I am...")
2. Keep it FOCUSED but informative (3-4 sentences)
3. Preserve all key attributes (role, expertise, resources, constraints, collaboration context)
4. Adjust goals and situation to fit the problem type
5. Maintain natural flow in the same language as the original

Output ONLY the rewritten persona, nothing else.

Rewritten persona:""",
            "persona_rewriting_chinese_simple": """你是一个将用户角色调整为适合询问特定问题类型的专家。

给定信息：
- 原始角色: {persona}
- 问题类型: "{problem_type}"

任务：用第三人称（描述"一个...的人"）重写这个角色，使其适合询问这个问题类型。保持简洁自然。

要求：
1. 使用第三人称描述（例如："一位研究员..." 而不是 "我是..."）
2. 保持简短聚焦（最多1-2句话）
3. 保留关键属性（角色、限制）
4. 调整情境以符合问题类型
5. 使用与原始角色相同的语言

只输出重写后的角色文本。

重写后的角色：""",
            "persona_rewriting_chinese_moderate": """你是一个将用户角色调整为适合询问特定问题类型的专家。

给定信息：
- 原始角色: {persona}
- 问题类型: "{problem_type}"

任务：用第三人称（描述"一个...的人"）重写这个角色，使其适合询问这个问题类型。使用适度的细节。

要求：
1. 使用第三人称描述（例如："一位教授..." 而不是 "我..."）
2. 保持简洁（2-3句话）
3. 保留核心特征（职业阶段、资源、工作环境）
4. 调整情况以与问题类型对齐
5. 使用与原始角色相同的语言自然书写

只输出重写后的角色文本。

重写后的角色：""",
            "persona_rewriting_chinese_detailed": """你是一个将用户角色调整为适合询问特定问题类型的专家。

给定信息：
- 原始角色: {persona}
- 问题类型: "{problem_type}"

任务：用第三人称（描述"一个...的人"）重写这个角色，使其适合询问这个问题类型。包含相关细节。

要求：
1. 使用第三人称描述（例如："一位资深研究员..." 而不是 "我是..."）
2. 保持聚焦但信息丰富（3-4句话）
3. 保留所有关键属性（角色、专长、资源、限制、合作环境）
4. 调整目标和情况以适应问题类型
5. 使用与原始角色相同的语言保持自然流畅

只输出重写后的角色文本。

重写后的角色：""",
            "query_generation_english": """You are an expert at generating diverse, high-quality queries for agent frameworks based on specific contexts.

Given:
- User Persona: {persona}
- Selected Problem Type: "{problem_type}"
{framework_description_block}{search_context_block}

Generate exactly 3 distinct queries with different difficulty levels (easy, medium, hard) that:
1. Are posed from the perspective of the given persona (as if this user is asking the question)
2. Are specifically designed for the selected problem type: "{problem_type}"
3. Are realistic and actionable
4. Vary significantly in complexity and scope
5. Are COMPLETE, self-contained tasks that can be executed from scratch without requiring any previously completed work or unavailable information
6. **Have DIVERSE questioning styles and formats** - avoid using the same sentence structure or phrasing pattern for all queries

⚠️ CRITICAL QUERY REQUIREMENTS:
- Each query must be a standalone task that provides ALL necessary context and information
- Do NOT create queries that depend on "previous analysis", "existing data", "prior work", or "already established" anything
- Do NOT use phrases like "continue the analysis", "build upon", "expand the previous", "update the existing"
- Each query should start fresh and include all required background information within the query itself
- All queries must be appropriate for the problem type: "{problem_type}"

🎨 STYLE DIVERSITY REQUIREMENTS:
- Use DIFFERENT questioning formats: direct questions
- Vary sentence structures: some queries can be short and direct, others can be detailed with context
- Mix formal and informal tones when appropriate to the persona

🚫 ABSOLUTELY FORBIDDEN - DO NOT include ANY of the following in your queries:
- Any framework names whatsoever
- Tag trace paths or arrow symbols (e.g., "A → B → C")
- Problem type labels as metadata tags (e.g., "(某某分析)", "(problem type: XXX)")
- Any references to "tag trace", "problem type context", or framework configuration
- Parenthetical labels showing classification

✅ YOUR QUERIES MUST BE:
- Natural questions that a real user would ask
- Free from any framework/system metadata
- Written as if the user knows nothing about the underlying classification system
- Stylistically diverse in both format and tone

⚠️ FORMAT REQUIREMENTS - Must strictly use asterisks:

**QUERIES:**
**EASY:** [Your easy query here]
**MEDIUM:** [Your medium query here]
**HARD:** [Your hard query here]

Note: You MUST use double asterisks ** around labels. Do not omit them!

Ensure each query is complete, specific, and appropriate for the given difficulty level and problem type.""",
            "query_generation_chinese": """你是一个专门为智能体框架生成多样化、高质量查询的专家，能够基于特定上下文生成相关的查询任务。

给定信息：
- 用户角色: {persona}
- 选定的问题类型: "{problem_type}"
{framework_description_block}{search_context_block}

请生成3个不同难度级别（简单、中等、困难）的截然不同的查询，这些查询需要：
1. 从给定用户的视角提出（就像这个用户在提问）
2. 专门针对选定的问题类型: "{problem_type}"
3. 切实可行且具有可操作性
4. 在复杂性和范围上有显著差异
5. 是完整、独立的任务，可以从零开始执行，无需依赖任何已完成的工作或不可获得的信息
6. **具有多样化的提问风格和格式** - 避免所有查询使用相同的句式结构或表达模式

⚠️ 查询关键要求：
- 每个查询必须是独立任务，提供所有必要的上下文和信息
- 不要创建依赖"之前的分析"、"现有数据"、"先前工作"或"已建立"内容的查询
- 不要使用"继续分析"、"基于"、"扩展之前的"、"更新现有的"等表述
- 每个查询都应从头开始，在查询本身内包含所有必需的背景信息
- 所有查询都必须适用于问题类型: "{problem_type}"

🎨 风格多样性要求：
- 使用不同的提问格式
- 变换句式结构：有的查询可以简短直接，有的可以详细附带上下文
- 根据角色特点适当混合正式和非正式语气

🚫 绝对禁止 - 查询中不得包含以下任何内容：
- 任何框架的名称
- 标签轨迹路径或箭头符号（如 "A → B → C"）
- 作为元数据标签的问题类型（如 "(某某分析)"、"(问题类型: XXX)"）
- 任何提及 "tag trace"、"标签轨迹"、"问题类型上下文" 或框架配置的内容
- 显示分类的括号标签

✅ 你的查询必须：
- 是真实用户会提出的自然问题
- 不含任何框架/系统元数据
- 就像用户完全不知道底层分类系统一样撰写
- 在格式和语气上具有风格多样性

⚠️ 格式要求 - 必须严格使用星号标记：

**查询任务:**
**简单:** [你的简单查询任务]
**中等:** [你的中等查询任务]
**困难:** [你的困难查询任务]

注意：必须使用两个星号 ** 包围标签，不要遗漏星号！

确保每个查询完整、具体，且适合给定的难度级别和问题类型。""",
            "file_requirement_english": """You are a classifier that determines whether a user query explicitly requires files provided by the system.\n\nQuery:\n{query}\n\nRespond strictly in JSON with the following schema:\n{{\n  \"requires_files\": <true or false>,\n  \"reason\": \"<short explanation>\",\n  \"required_items\": [\"<item description>\", ...]\n}}\n\nMark requires_files as true only when the user clearly expects system-provided attachments or files. Otherwise, return false and an empty list.""",
            "file_requirement_chinese": """你是一名判断查询是否明确需要系统提供文件的分类器。\n\n查询：\n{query}\n\n请严格按照以下 JSON 模式输出：\n{{\n  \"requires_files\": <true 或 false>,\n  \"reason\": \"<简短说明>\",\n  \"required_items\": [\"<文件需求描述>\", ...]\n}}\n\n只有在查询明确要求系统附带文件时，才将 requires_files 设为 true；否则返回 false 和空列表。""",
            "file_plan_english": """You are planning support files for an assistant. The assistant will provide local files when necessary.\n\nQuery:\n{query}\nRequested items (if any):\n{required_items}\n\nReturn a JSON object:{{\n  \"directory_name\": \"<short descriptive slug for the download folder>\",\n  \"files\": [\n    {{\"description\": \"<what the file contains>\", \"url\": \"<direct HTTPS download URL>\"}},\n    ...\n  ]\n}}\n\nSTRICT REQUIREMENTS:\n- Supply directory_name as a concise kebab-case summary (letters, numbers, hyphen only).\n- Every \"url\" MUST start with https:// and be publicly accessible.\n- Do NOT use http:// even if it exists; omit entries without an https variant.\n- Do NOT return data:, file:, ftp:, or any other scheme.\n- Omit entries you cannot satisfy.\n- Use an empty list if no files are required.""",
            "file_plan_chinese": """你负责为助手规划需要提供的支持文件。\n\n查询：\n{query}\n若有明确需求：\n{required_items}\n\n请返回 JSON：{{\n  \"directory_name\": \"<下载文件夹的简短说明（kebab-case）>\",\n  \"files\": [\n    {{\"description\": \"<文件说明>\", \"url\": \"<可直接下载的 HTTPS 链接>\"}},\n    ...\n  ]\n}}\n\n严格要求：\n- directory_name 请使用简洁的连字符命名（只含字母/数字/连字符）。\n- 每个 url 必须以 https:// 开头且可公开访问。\n- 禁止使用 http://；如果没有 https 版本，请省略该条目。\n- 禁止返回 data:、file:、ftp: 等协议。\n- 若无法满足，请省略条目；若无需文件则返回空列表。""",
            "file_rewrite_english": """You are enhancing a user query so it naturally references local files provided by the system.\n\nOriginal query:\n{query}\n\nAvailable local files:\n{files}\n\nWrite a rewritten query that:\n1. Remains faithful to the user's intent.\n2. References the provided file paths exactly as shown.\n3. Reads naturally in English.\n\nReturn JSON: {{\"rewritten_query\": \"<updated query>\"}}""",
            "file_rewrite_chinese": """你要改写查询，使其自然引用系统提供的本地文件。\n\n原始查询：\n{query}\n可用文件：\n{files}\n\n请生成一个保持原意、自然提及这些文件路径的中文查询。返回 JSON：{{\"rewritten_query\": \"<改写后的查询>\"}}""",
            "fuzzify_query": """You are an expert editor who makes structured queries feel more like natural requests from humans\nwho assume shared context. Your job is to keep the user’s intent intact while softening how explicitly\nit is spelled out.\n\nWORKFLOW:\n\t1.\tTake in the query fully — understand not just what it asks, but the voice behind it, the rhythm it follows, and the direction it leans toward.\n\t2.\tNotice where human intuition would fill the gaps instead of spelling out every point. Those are the places to smooth and simplify.\n\t3.\tLet the rewrite breathe: merge rigid bullets into an easy flow, as if explaining to a peer. Keep the same spirit and order, but let details blur into gestures或含蓄表达。Avoid invention; reveal less while implying more.\n\t4.\tEven when trimming or veiling specifics, make sure the reasoning thread stays intact beneath the surface — readable as natural language, yet still rich enough for an LLM to intuit the hidden structure or logic when needed.\n\t5.\tFor concrete requirements or illustrative examples,你可以在这些信息对读者来说属于“约定俗成”或“可推断”的情况下，将其模糊或省略；但不要删掉让指令失去连贯性的关键约束。\n\t6.\tPresent the final output as a JSON block of the form {{\"analysis\": \"...\", \"fuzzy_query\": \"...\"}} — concise, no extras.\n\t\t- analysis should clearly describe why and how the text was softened or restructured — what was condensed, implied, or reframed.\n\t\t- fuzzy_query should capture the final, natural version of the query after applying the workflow — fluent, human-sounding, and implicitly carrying the logic of the original.\n\nRESPONSE RULES:\n- analysis: 用一到两句话说明原始意图，并明确指出哪些信息被折叠为“默认背景”或“常识”，以及这样做的原因（口语化、弱化细节等）。\n- fuzzy_query: 生成与原文语气和结构一致的版本，但在身份、约束、资源、示例等处，通过口语化、省略、指代或语气缓冲等方式，营造自然的“真实用户式”模糊；核心目标与逻辑须完整保留。\n- 可保留原有段落或条目结构，也可在合理情况下改写为说明性语段；避免引入新的数值、设定或削弱关键目的。\n- 最终输出必须为单行 JSON，格式严格为 {{\"analysis\": \"...\", \"fuzzy_query\": \"...\"}}，不得包含多余文本或 Markdown。\n\nORIGINAL QUERY:\n{query}\n\nReturn ONLY the JSON object.""",
        }

        # 为了向后兼容，保留旧的键名
        self.templates["query_generation"] = self.templates["query_generation_english"]

    def format_file_requirement_prompt(
        self, query: str, language: str = "english"
    ) -> str:
        template_key = f"file_requirement_{language.lower()}"
        if template_key not in self.templates:
            template_key = "file_requirement_english"
        return self.get_template(template_key).format(query=query)

    def format_file_system_plan_prompt(
        self, query: str, required_items: List[str], language: str = "english"
    ) -> str:
        template_key = f"file_plan_{language.lower()}"
        if template_key not in self.templates:
            template_key = "file_plan_english"
        required_text = (
            "\n".join(f"- {item}" for item in required_items)
            if required_items
            else "(none)"
        )
        return self.get_template(template_key).format(
            query=query, required_items=required_text
        )

    def format_file_query_rewrite_prompt(
        self, query: str, files: List[Dict[str, str]], language: str = "english"
    ) -> str:
        template_key = f"file_rewrite_{language.lower()}"
        if template_key not in self.templates:
            template_key = "file_rewrite_english"
        formatted_files = "\n".join(
            f"- {file_info.get('description', '').strip() or 'File'}: {file_info.get('local_path')}"
            for file_info in files
            if file_info.get("local_path")
        )
        return self.get_template(template_key).format(
            query=query, files=formatted_files
        )

    def format_fuzzifier_prompt(self, query: str) -> str:
        return self.get_template("fuzzify_query").format(query=query)

    def _format_framework_description_block(
        self, framework_description: Optional[str], language: str
    ) -> str:
        """Format framework description block for inclusion in prompt."""
        if not framework_description:
            return ""

        if language.lower() == "chinese":
            return f"""

**⚠️ 框架适配性要求（重要）:**
生成的查询必须严格参考下述框架描述，确保查询内容适合该框架解决。生成的每个查询都应该在框架的能力范围内，不要生成超出框架能力范围的任务。

- 框架描述: {framework_description}"""
        else:
            return f"""

**⚠️ FRAMEWORK SUITABILITY REQUIREMENT (IMPORTANT):**
The generated queries MUST strictly reference the framework description provided below and ensure that the query content is suitable for this framework to solve. Every generated query should be within the framework's capability scope - do NOT generate tasks that are beyond the framework's abilities.

- Framework Description: {framework_description}"""

    def _format_search_context_block(
        self, search_context: Optional[Dict[str, Any]], language: str
    ) -> str:
        if not search_context:
            return ""

        results = search_context.get("results") or []
        queries = search_context.get("queries") or []
        if not results and not queries:
            return ""

        max_items = 5
        if language.lower() == "chinese":
            lines = ["- 最新外部检索摘要："]
            if queries:
                lines.append("  检索词：" + "；".join(queries[:3]))
            for idx, item in enumerate(results[:max_items], start=1):
                title = item.get("title") or "(无标题)"
                source = item.get("source") or ""
                date = item.get("date") or ""
                snippet = item.get("snippet") or ""
                details = title
                extras = " ".join(filter(None, [source, date])).strip()
                if extras:
                    details += f"（来源：{extras}）"
                lines.append(f"  {idx}. {details}")
                if snippet:
                    lines.append(f"     摘要：{snippet}")
            return "\n".join(lines)

        # Default to English
        lines = ["- Recent Findings from Web Search:"]
        if queries:
            lines.append("  Queries: " + ", ".join(queries[:3]))
        for idx, item in enumerate(results[:max_items], start=1):
            title = item.get("title") or "(no title)"
            source = item.get("source") or ""
            date = item.get("date") or ""
            snippet = item.get("snippet") or ""
            descriptor = title
            extras = " ".join(filter(None, [source, date])).strip()
            if extras:
                descriptor += f" — {extras}"
            lines.append(f"  {idx}. {descriptor}")
            if snippet:
                lines.append(f"     Summary: {snippet}")
        return "\n".join(lines)

    def get_template(self, template_name: str) -> str:
        """Get a prompt template by name"""
        return self.templates.get(template_name, "")

    def format_query_generation_prompt(
        self,
        persona: FrameworkPersona,
        tag_trace: TagTrace,
        problem_type: str,
        language: str = "english",
        search_context: Optional[Dict[str, Any]] = None,
        framework_description: Optional[str] = None,
    ) -> str:
        """Format the query generation prompt with specific context

        Args:
            persona: The framework persona
            tag_trace: TagTrace object (retained for backward compatibility but not used in prompt)
            problem_type: The specific problem type selected from the trace
            language: Language for query generation ("english" or "chinese")
            search_context: Optional web search context
            framework_description: Optional framework description from framework_config.yaml
        """
        # Select appropriate template based on language
        template_key = f"query_generation_{language.lower()}"

        # Fallback to English if language not supported
        if template_key not in self.templates:
            template_key = "query_generation_english"

        framework_desc_block = self._format_framework_description_block(
            framework_description, language
        )
        search_block = self._format_search_context_block(search_context, language)

        return self.get_template(template_key).format(
            persona=persona.get_persona(language),
            problem_type=problem_type,
            framework_description_block=framework_desc_block,
            search_context_block=search_block,
        )

    def format_persona_evaluation_prompt(
        self, persona: str, problem_type: str, language: str = "english"
    ) -> str:
        """Format the persona evaluation prompt

        Args:
            persona: The persona text to evaluate
            problem_type: The problem type to evaluate against
            language: Language for evaluation ("english" or "chinese")
        """
        template_key = f"persona_evaluation_{language.lower()}"
        if template_key not in self.templates:
            template_key = "persona_evaluation_english"

        return self.get_template(template_key).format(
            persona=persona, problem_type=problem_type
        )

    def format_persona_rewriting_prompt(
        self, persona: str, problem_type: str, language: str = "english"
    ) -> str:
        """Format the persona rewriting prompt with random detail level

        Args:
            persona: The original persona text to rewrite
            problem_type: The problem type to adapt for
            language: Language for rewriting ("english" or "chinese")
        """
        # Randomly select detail level: simple, moderate, or detailed
        import random

        detail_levels = ["simple", "moderate", "detailed"]
        detail_level = random.choice(detail_levels)

        template_key = f"persona_rewriting_{language.lower()}_{detail_level}"

        # Fallback to moderate if specific template not found
        if template_key not in self.templates:
            template_key = f"persona_rewriting_{language.lower()}_moderate"

        # Final fallback
        if template_key not in self.templates:
            template_key = "persona_rewriting_english_moderate"

        logger.info(f"Selected persona rewriting detail level: {detail_level}")

        return self.get_template(template_key).format(
            persona=persona, problem_type=problem_type
        )


class ResponseParser:
    """
    Parses LLM responses to extract structured data.
    """

    @staticmethod
    def parse_query_generation_response(
        response: str, language: str = "english"
    ) -> List[GeneratedQuery]:
        """
        Parse the query generation response to extract queries.

        Args:
            response: The LLM response text
            language: Language of the response ("english" or "chinese")
        """
        queries = []

        if language.lower() == "chinese":
            # Parse Chinese format
            queries = ResponseParser._parse_chinese_queries(response)
        else:
            # Parse English format (default)
            queries = ResponseParser._parse_english_queries(response)

        return queries

    @staticmethod
    def _parse_english_queries(response: str) -> List[GeneratedQuery]:
        """Parse English format queries"""
        queries = []

        # Extract queries
        query_pattern = r"\*\*(\w+):\*\*\s*(.*?)(?=\*\*\w+:\*\*|$)"
        query_matches = re.findall(query_pattern, response, re.DOTALL)

        # Debug: log if no matches found
        if not query_matches:
            logger.warning(
                f"No query matches found in response. Response preview: {response[:500]}..."
            )

        for difficulty, content in query_matches:
            if difficulty.upper() in ["EASY", "MEDIUM", "HARD"]:
                query = GeneratedQuery(
                    content=content.strip(),
                    difficulty=difficulty.lower(),
                    context={},
                    metadata={},
                )
                queries.append(query)

        return queries

    @staticmethod
    def _parse_chinese_queries(response: str) -> List[GeneratedQuery]:
        """Parse Chinese format queries"""
        queries = []

        # Extract Chinese queries - pattern for "**简单:**", "**中等:**", "**困难:**"
        query_pattern = (
            r"\*\*(简单|中等|困难):\*\*\s*(.*?)(?=\*\*(?:简单|中等|困难):\*\*|$)"
        )
        query_matches = re.findall(query_pattern, response, re.DOTALL)

        # Debug: log if no matches found
        if not query_matches:
            logger.warning(
                f"No Chinese query matches found in response. Response preview: {response[:500]}..."
            )

        # Mapping Chinese difficulty to English
        difficulty_map = {"简单": "easy", "中等": "medium", "困难": "hard"}

        for difficulty_chinese, content in query_matches:
            difficulty_english = difficulty_map.get(difficulty_chinese, "unknown")
            query = GeneratedQuery(
                content=content.strip(),
                difficulty=difficulty_english,
                context={},
                metadata={},
            )
            queries.append(query)

        return queries


class QueryGenerator:
    """
    Main query generation system that coordinates LLM calls, prompt formatting,
    and response parsing.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        web_search_config: Optional[Dict[str, object]] = None,
    ):
        self.llm_client = llm_client
        self.prompt_manager = PromptTemplateManager()
        self.response_parser = ResponseParser()
        self.rewrite_agent = RewriteAgent(self.llm_client, self.prompt_manager)
        self.query_agent = QuerySynthesisAgent(
            llm_client=self.llm_client,
            prompt_manager=self.prompt_manager,
            response_parser=self.response_parser,
            generated_query_cls=GeneratedQuery,
            result_cls=QueryGenerationResult,
        )
        self.file_requirement_agent = FileRequirementAgent(
            self.llm_client, self.prompt_manager
        )
        self.file_system_agent = FileSystemAgent(
            llm_client=self.llm_client,
            prompt_manager=self.prompt_manager,
            base_dir=Path(__file__).parent.parent / "file_system",
        )
        self.file_augmentation_agent = FileAugmentationAgent(
            self.llm_client, self.prompt_manager
        )
        self.web_research_agent: Optional[WebResearchAgent] = None
        self.fuzzifier_agent = FuzzifierAgent(
            llm_client=self.llm_client,
            prompt_manager=self.prompt_manager,
            probability=0.0,
        )
        self.url_processing_agent = URLProcessingAgent(
            llm_client=self.llm_client, max_repair_attempts=3
        )
        self.router_agent = RouterAgent(
            rewrite_agent=self.rewrite_agent,
            web_research_agent=self.web_research_agent,
            query_agent=self.query_agent,
            requirement_agent=self.file_requirement_agent,
            file_system_agent=self.file_system_agent,
            file_augmentation_agent=self.file_augmentation_agent,
            fuzzifier_agent=self.fuzzifier_agent,
            url_processing_agent=self.url_processing_agent,
        )
        self.set_web_search_config(web_search_config)

    def set_web_search_config(self, config: Optional[Dict[str, object]]) -> None:
        """Update the web search configuration and wire the agent into the router."""
        config = config or {}
        probability = float(config.get("probability", 0.0) or 0.0)
        api_key = (
            config.get("api_key") or config.get("apiKey") or os.getenv("SERPER_API_KEY")
        )
        config.setdefault("api_key", api_key)
        config["probability"] = probability

        if probability > 0 and api_key:
            if self.web_research_agent is None:
                self.web_research_agent = WebResearchAgent(config)
            else:
                self.web_research_agent.update_config(config)
        else:
            self.web_research_agent = None

        self.router_agent.set_web_research_agent(self.web_research_agent)

    def set_fuzzifier_probability(self, probability: float) -> None:
        """Configure the fuzzifier subagent invocation probability."""
        if self.fuzzifier_agent:
            self.fuzzifier_agent.set_probability(probability)
            self.router_agent.set_fuzzifier_agent(self.fuzzifier_agent)

    def set_file_analysis_enabled(self, enabled: bool) -> None:
        """Configure whether file requirement analysis and file downloading are enabled."""
        self.router_agent.set_file_analysis_enabled(enabled)

    def set_url_processing_enabled(self, enabled: bool) -> None:
        """Configure whether URL processing (extraction, validation, repair) is enabled."""
        self.router_agent.set_url_processing_enabled(enabled)

    def _build_error_result(
        self,
        tag_trace: Optional[TagTrace],
        problem_type: str,
        language: str,
        framework_name: Optional[str],
        error_message: str,
    ) -> QueryGenerationResult:
        """Create a standardized error result for downstream consumers."""
        trace_context: List[str] = []
        if tag_trace and hasattr(tag_trace, "get_labels"):
            trace_context = tag_trace.get_labels()

        metadata = {
            "error": error_message,
            "language": language,
            "timestamp": time.time(),
        }
        if framework_name:
            metadata["framework"] = framework_name

        return QueryGenerationResult(
            queries=[],
            trace_context=trace_context,
            problem_type=problem_type,
            generation_metadata=metadata,
        )

    def generate_queries(
        self,
        persona: FrameworkPersona,
        tag_trace: TagTrace,
        problem_type: str,
        language: str = "english",
        debug_mode: bool = False,
        difficulty_distribution: Optional[Dict[str, float]] = None,
        framework_name: str = None,
        framework_description: Optional[str] = None,
    ) -> QueryGenerationResult:
        """
        Generate queries by running the configured agent workflow.
        """
        context = AgentContext(
            data={
                "persona": persona,
                "problem_type": problem_type,
                "tag_trace": tag_trace,
                "language": language,
                "debug_mode": debug_mode,
                "difficulty_distribution": difficulty_distribution,
                "framework_name": framework_name,
                "framework_description": framework_description,
            }
        )

        router_output = self.router_agent.run(context)

        if router_output.timings:
            for key, value in router_output.timings.items():
                context.add_timing(f"router.{key}", value)

        if router_output.errors:
            for error in router_output.errors:
                context.append_error(error)

        if router_output.data:
            context.update(router_output.data)

        if not router_output.success:
            error_message = "; ".join(context.errors) or "router agent failed"
            logger.error("Router agent failed: %s", error_message)
            return self._build_error_result(
                tag_trace=tag_trace,
                problem_type=problem_type,
                language=language,
                framework_name=framework_name,
                error_message=error_message,
            )

        if context.errors:
            error_message = "; ".join(context.errors)
            logger.error("Agent pipeline failed: %s", error_message)
            return self._build_error_result(
                tag_trace=tag_trace,
                problem_type=problem_type,
                language=language,
                framework_name=framework_name,
                error_message=error_message,
            )

        result: Optional[QueryGenerationResult] = context.get("query_result")
        if result is None:
            logger.error("Query synthesis agent did not return a result")
            return self._build_error_result(
                tag_trace=tag_trace,
                problem_type=problem_type,
                language=language,
                framework_name=framework_name,
                error_message="query_result missing from agent context",
            )

        metadata = result.generation_metadata
        metadata["timings"] = dict(context.timings)
        metadata.setdefault("language", language)
        metadata.setdefault("problem_type", problem_type)
        if framework_name and "framework" not in metadata:
            metadata["framework"] = framework_name

        metadata["persona_was_rewritten"] = context.get("persona_was_rewritten", False)

        search_context = context.get("search_context")
        if search_context:
            metadata["search_context"] = search_context

        downloaded_files = context.get("all_downloaded_files")
        if downloaded_files:
            file_system_info = metadata.setdefault("file_system", {})
            file_system_info["downloads"] = downloaded_files

        if context.get("persona_was_rewritten"):
            rewritten_persona = context.get("rewritten_persona")
            if isinstance(rewritten_persona, FrameworkPersona):
                metadata.setdefault("rewritten_persona", rewritten_persona.persona)
                if rewritten_persona.persona_chinese:
                    metadata.setdefault(
                        "rewritten_persona_chinese", rewritten_persona.persona_chinese
                    )

        logger.info(
            "Query generation completed with %d query(ies)", len(result.queries)
        )

        return result

    def batch_generate_queries(
        self, generation_requests: List[Dict[str, Any]], language: str = "english"
    ) -> List[QueryGenerationResult]:
        """
        Generate queries for multiple requests in batch mode.
        """
        results: List[QueryGenerationResult] = []

        for request in generation_requests:
            try:
                request_language = request.get("language", language)
                result = self.generate_queries(
                    persona=request["persona"],
                    tag_trace=request["tag_trace"],
                    problem_type=request["problem_type"],
                    language=request_language,
                    debug_mode=request.get("debug_mode", False),
                    difficulty_distribution=request.get("difficulty_distribution"),
                    framework_name=request.get("framework_name"),
                )
                results.append(result)
                time.sleep(0.1)  # Avoid rate limiting

            except Exception as exc:  # noqa: BLE001
                logger.error("Error in batch generation: %s", exc)
                results.append(
                    self._build_error_result(
                        tag_trace=request.get("tag_trace"),
                        problem_type=request.get("problem_type", ""),
                        language=request.get("language", language),
                        framework_name=request.get("framework_name"),
                        error_message=str(exc),
                    )
                )

        return results


class QueryExporter:
    """
    Exports generated queries to various formats.
    """

    @staticmethod
    def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Strip internal-only fuzzifier fields before export."""
        if not metadata:
            return {}

        sanitized = copy.deepcopy(metadata)
        fuzz_meta = sanitized.get("fuzzifier")
        if isinstance(fuzz_meta, dict):
            fuzz_meta.pop("original_query", None)
        return sanitized

    @staticmethod
    def export_to_jsonl(
        results: List[QueryGenerationResult],
        output_file: Path,
        framework_name: str = None,
    ):
        """
        Export query generation results to JSONL format.

        Args:
            results: List of QueryGenerationResult objects
            output_file: Path to output JSONL file
            framework_name: Framework name to include in export (optional)
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                for query in result.queries:
                    query_metadata = query.metadata or {}
                    fuzz_meta = query_metadata.get("fuzzifier", {})
                    sanitized_metadata = QueryExporter._sanitize_metadata(
                        query_metadata
                    )
                    combined_metadata = {
                        **sanitized_metadata,
                        **(result.generation_metadata or {}),
                    }

                    export_data = {
                        "query": query.content,
                        "difficulty": query.difficulty,
                        "trace_context": result.trace_context,
                        "problem_type": result.problem_type,
                        "requires_local_files": query_metadata.get(
                            "requires_local_files", False
                        ),
                        "used_web_search": query_metadata.get("used_web_search", False),
                        "fuzzified": bool(fuzz_meta.get("applied")),
                        "metadata": combined_metadata,
                    }
                    if fuzz_meta.get("applied") and fuzz_meta.get("original_query"):
                        export_data["original_query"] = fuzz_meta.get("original_query")
                    # Add framework field if provided
                    if framework_name:
                        export_data["framework"] = framework_name
                    f.write(json.dumps(export_data, ensure_ascii=False) + "\n")

    @staticmethod
    def export_to_json(
        results: List[QueryGenerationResult],
        output_file: Path,
        framework_name: str = None,
    ):
        """
        Export query generation results to JSON format.

        Args:
            results: List of QueryGenerationResult objects
            output_file: Path to output JSON file
            framework_name: Framework name to include in export (optional)
        """
        export_data = []

        for result in results:
            queries_payload = []
            for query in result.queries:
                query_metadata = query.metadata or {}
                fuzz_meta = query_metadata.get("fuzzifier", {})
                sanitized_metadata = QueryExporter._sanitize_metadata(query_metadata)
                entry = {
                    "content": query.content,
                    "difficulty": query.difficulty,
                    "context": query.context,
                    "requires_local_files": query_metadata.get(
                        "requires_local_files", False
                    ),
                    "used_web_search": query_metadata.get("used_web_search", False),
                    "fuzzified": bool(fuzz_meta.get("applied")),
                    "metadata": sanitized_metadata,
                }
                if fuzz_meta.get("applied") and fuzz_meta.get("original_query"):
                    entry["original_query"] = fuzz_meta.get("original_query")
                queries_payload.append(entry)

            result_data = {
                "queries": queries_payload,
                "problem_type": result.problem_type,
                "trace_context": result.trace_context,
                "generation_metadata": result.generation_metadata,
            }
            # Add framework field if provided
            if framework_name:
                result_data["framework"] = framework_name
            export_data.append(result_data)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
