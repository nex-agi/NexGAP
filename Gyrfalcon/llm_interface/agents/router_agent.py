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
Router agent that coordinates rewrite, query generation, and optional file provisioning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from frameworks.framework_manager import FrameworkPersona

if TYPE_CHECKING:
    from llm_interface.query_generator import QueryGenerationResult

from .base import Agent, AgentContext, AgentOutput

logger = logging.getLogger(__name__)


class RouterAgent(Agent):
    """High-level controller for the multi-agent query synthesis workflow."""

    def __init__(
        self,
        rewrite_agent: Agent,
        query_agent: Agent,
        requirement_agent: Agent,
        file_system_agent: Agent,
        file_augmentation_agent: Agent,
        web_research_agent: Optional[Agent] = None,
        fuzzifier_agent: Optional[Agent] = None,
        url_processing_agent: Optional[Agent] = None,
    ):
        super().__init__("router")
        self.rewrite_agent = rewrite_agent
        self.query_agent = query_agent
        self.requirement_agent = requirement_agent
        self.file_system_agent = file_system_agent
        self.file_augmentation_agent = file_augmentation_agent
        self.web_research_agent = web_research_agent
        self.fuzzifier_agent = fuzzifier_agent
        self.url_processing_agent = url_processing_agent
        self.file_analysis_enabled = False  # Default: file analysis disabled
        self.url_processing_enabled = False  # Default: URL processing disabled

    def set_web_research_agent(self, agent: Optional[Agent]) -> None:
        self.web_research_agent = agent

    def set_fuzzifier_agent(self, agent: Optional[Agent]) -> None:
        self.fuzzifier_agent = agent

    def set_url_processing_agent(self, agent: Optional[Agent]) -> None:
        """Set or update the URL processing agent."""
        self.url_processing_agent = agent

    def set_file_analysis_enabled(self, enabled: bool) -> None:
        """Enable or disable file requirement analysis and file downloading."""
        self.file_analysis_enabled = enabled
        logger.info(f"File analysis {'enabled' if enabled else 'disabled'}")

    def set_url_processing_enabled(self, enabled: bool) -> None:
        """Enable or disable URL processing (extraction, validation, repair)."""
        self.url_processing_enabled = enabled
        logger.info(f"URL processing {'enabled' if enabled else 'disabled'}")

    def run(self, context: AgentContext) -> AgentOutput:
        if not isinstance(context.get("persona"), FrameworkPersona):
            error = "RouterAgent requires FrameworkPersona in context."
            logger.error(error)
            return AgentOutput(success=False, errors=[error])

        # Persona rewrite
        if not self._execute_agent(self.rewrite_agent, context):
            return AgentOutput(success=False, errors=context.errors)

        if self.web_research_agent:
            if not self._execute_agent(self.web_research_agent, context):
                return AgentOutput(success=False, errors=context.errors)

        # Query synthesis
        if not self._execute_agent(self.query_agent, context):
            return AgentOutput(success=False, errors=context.errors)

        result: Optional["QueryGenerationResult"] = context.get("query_result")
        if result is None:
            error = "Query synthesis agent did not produce a result."
            logger.error(error)
            return AgentOutput(success=False, errors=[error])

        # Process each query for URL processing and file requirements
        updated_queries = []
        all_downloads: List[Dict[str, str]] = []

        search_context = context.get("search_context")

        for query in result.queries:
            if query is None:
                continue

            context.set("current_query_object", query)
            context.set("current_query_text", query.content)
            context.set("query_requires_files", False)
            context.set("downloaded_files", [])
            context.set("augmented_query_text", None)
            context.set("query_required_items", [])
            context.set("file_metadata_update", None)

            # URL Processing - Process URLs in the query if enabled
            if self.url_processing_enabled and self.url_processing_agent:
                logger.info(f"🔗 Processing URLs in query...")
                url_context = AgentContext(
                    data={
                        "query": query.content,
                        "language": context.get("language", "english"),
                    }
                )

                url_output = self.url_processing_agent.run(url_context)

                # Update context with output data (same as _execute_agent does)
                if url_output.data:
                    url_context.update(url_output.data)

                if url_output.success:
                    url_processing_result = url_context.get("url_processing")
                    processed_query = url_context.get("processed_query", query.content)
                    url_stats = url_context.get("url_stats", {})

                    # Update query content if URLs were processed
                    if processed_query != query.content:
                        logger.info("✅ Query updated with URL changes")
                        query.content = processed_query

                    # Add URL processing results to metadata
                    if url_processing_result:
                        query.metadata["url_processing"] = url_processing_result

                    # Add URL statistics
                    if url_stats:
                        query.metadata["url_stats"] = url_stats

                        # Log statistics
                        if url_stats.get("total_extracted", 0) > 0:
                            logger.info(
                                f"   📊 URLs: {url_stats['total_extracted']} extracted, "
                                f"{url_stats.get('accessible', 0)} accessible, "
                                f"{url_stats.get('repaired', 0)} repaired, "
                                f"{url_stats.get('removed', 0)} removed"
                            )
                else:
                    logger.warning(f"⚠️ URL processing failed: {url_output.errors}")

            # Only perform file requirement analysis if enabled
            if self.file_analysis_enabled:
                if not self._execute_agent(self.requirement_agent, context):
                    continue

                requires_files = context.get("query_requires_files", False)
                requirement_reason = context.get(
                    "query_file_requirement_reason", ""
                ).strip()
                required_items = context.get("query_required_items", []) or []

                if requires_files:
                    self._execute_agent(self.file_system_agent, context)
                    downloaded_files = context.get("downloaded_files", [])
                    all_downloads.extend(downloaded_files)
                    self._execute_agent(self.file_augmentation_agent, context)

                    augmented_text = context.get("augmented_query_text")
                    metadata_update = context.get("file_metadata_update") or {}

                    if augmented_text:
                        query.content = augmented_text

                    if metadata_update:
                        query.metadata = {**query.metadata, **metadata_update}
                else:
                    context.set("downloaded_files", [])
            else:
                # File analysis disabled - skip file-related agents
                requires_files = False
                requirement_reason = ""
                required_items = []
                context.set("downloaded_files", [])

            if self.fuzzifier_agent:
                context.set("current_query_object", query)
                context.set("current_query_text", query.content)

                # Log fuzzifier agent invocation with status indicator
                fuzz_probability = getattr(self.fuzzifier_agent, "probability", 0.0)
                logger.info(
                    f"🔮 Invoking fuzzifier agent (probability: {fuzz_probability:.2f})"
                )

                self._execute_agent(self.fuzzifier_agent, context)
                updated_obj = context.get("current_query_object")
                if updated_obj is not None:
                    query = updated_obj

                    # Check if fuzzification was actually applied
                    fuzz_meta = (
                        query.metadata.get("fuzzifier", {}) if query.metadata else {}
                    )
                    if fuzz_meta.get("applied"):
                        logger.info("✅ Query fuzzified successfully")
                    elif fuzz_meta.get("attempted"):
                        logger.info("⚠️ Fuzzifier attempted but not applied")
                    else:
                        logger.debug(
                            "🔄 Fuzzifier skipped (probability gate or no query text)"
                        )

            # Ensure metadata records the file requirement outcome
            metadata_snapshot = {**query.metadata}
            metadata_snapshot["requires_local_files"] = bool(requires_files)
            if requires_files:
                if requirement_reason:
                    metadata_snapshot["file_requirement_reason"] = requirement_reason
                else:
                    metadata_snapshot.pop("file_requirement_reason", None)
                if required_items:
                    metadata_snapshot["file_required_items"] = required_items
                else:
                    metadata_snapshot.pop("file_required_items", None)
            else:
                metadata_snapshot.pop("file_requirement_reason", None)
                metadata_snapshot.pop("file_required_items", None)
                metadata_snapshot.pop("file_system", None)  # remove stale file info
            metadata_snapshot["used_web_search"] = bool(
                search_context and search_context.get("used")
            )
            query.metadata = metadata_snapshot

            updated_queries.append(query)

        # Collect metadata
        if all_downloads:
            context.set("all_downloaded_files", all_downloads)

        context.set("query_result", result)

        return AgentOutput(success=True, data={"query_result": result})

    def _execute_agent(self, agent: Agent, context: AgentContext) -> bool:
        output = agent.run(context)

        if output.timings:
            for key, value in output.timings.items():
                context.add_timing(f"{agent.name}.{key}", value)

        if output.errors:
            for error in output.errors:
                context.append_error(f"{agent.name}: {error}")

        if not output.success:
            logger.error("Agent %s failed: %s", agent.name, output.errors)
            return False

        if output.data:
            context.update(output.data)

        return True
