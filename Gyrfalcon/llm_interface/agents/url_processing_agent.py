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
URL Processing Orchestrator Agent for Gyrfalcon v5

Coordinates the complete URL processing pipeline:
1. Extract URLs from query
2. Validate URLs
3. Repair broken URLs (with iterative retry)
4. Rewrite query with fixed/removed URLs
"""

import logging
import time
from typing import Any, Dict, List

from .base import Agent, AgentContext, AgentOutput
from .url_extraction_agent import URLExtractionAgent
from .url_query_rewrite_agent import URLQueryRewriteAgent
from .url_repair_agent import URLRepairAgent
from .url_validator_agent import URLValidatorAgent

logger = logging.getLogger(__name__)


class URLProcessingAgent(Agent):
    """
    Orchestrates the complete URL processing pipeline.
    Includes iterative repair with multiple attempts.
    """

    def __init__(
        self, llm_client, max_repair_attempts: int = 3, name: str = "URLProcessingAgent"
    ):
        super().__init__(name)
        self.llm_client = llm_client
        self.max_repair_attempts = max_repair_attempts

        # Initialize sub-agents
        self.extraction_agent = URLExtractionAgent(llm_client)
        self.validator = URLValidatorAgent()
        self.repair_agent = URLRepairAgent(llm_client)
        self.rewrite_agent = URLQueryRewriteAgent(llm_client)

    def run(self, context: AgentContext) -> AgentOutput:
        """
        Execute the complete URL processing pipeline.

        Expected context keys:
            - query: str (the query to process)
            - language: str (optional, "english" or "chinese")

        Adds to context:
            - url_processing: Dict (complete URL processing results)
            - processed_query: str (final query after URL processing)
        """
        start_time = time.time()

        query = context.get("query")
        language = context.get("language", "english")

        if not query:
            return AgentOutput(
                success=False, errors=["No query provided for URL processing"]
            )

        try:
            logger.info("Starting URL processing pipeline...")

            # Step 1: Extract URLs
            logger.info("\n[Step 1] Extracting URLs...")
            extraction_ctx = AgentContext(data={"query": query, "language": language})
            extraction_output = self.extraction_agent.run(extraction_ctx)

            if not extraction_output.success:
                return AgentOutput(success=False, errors=extraction_output.errors)

            # Transfer output data to context (same pattern as RouterAgent)
            if extraction_output.data:
                extraction_ctx.update(extraction_output.data)

            extracted_urls = extraction_ctx.get("extracted_urls", [])
            has_urls = extraction_ctx.get("has_urls", False)

            logger.info(f"  Found {len(extracted_urls)} URL(s)")

            # If no URLs, return original query
            if not has_urls:
                logger.info("No URLs found, skipping URL processing")
                return AgentOutput(
                    success=True,
                    data={
                        "url_processing": {
                            "extraction": {"has_urls": False, "urls": []}
                        },
                        "processed_query": query,
                    },
                    timings={"url_processing": time.time() - start_time},
                )

            # Step 2: Validate URLs
            logger.info("\n[Step 2] Validating URLs...")
            validation_ctx = AgentContext(data={"extracted_urls": extracted_urls})
            validation_output = self.validator.run(validation_ctx)

            if not validation_output.success:
                return AgentOutput(success=False, errors=validation_output.errors)

            # Transfer output data to context
            if validation_output.data:
                validation_ctx.update(validation_output.data)

            validation_results = validation_ctx.get("validation_results", [])
            accessible_count = validation_ctx.get("accessible_count", 0)
            broken_urls = validation_ctx.get("broken_urls", [])

            logger.info(f"  Accessible: {accessible_count}/{len(extracted_urls)}")

            # Step 3: Repair broken URLs (iterative)
            repair_results = []
            url_changes = []

            if broken_urls:
                logger.info(f"\n[Step 3] Repairing {len(broken_urls)} broken URL(s)...")

                for url_info in broken_urls:
                    original_url = url_info["url"]
                    logger.info(f"\n  Repairing: {original_url}")
                    logger.info(f"  Initial error: {url_info['error']}")

                    # Track all repair attempts for this URL
                    repair_attempts = []
                    working_url = None
                    current_error = url_info["error"]

                    # Loop through repair attempts
                    for attempt in range(self.max_repair_attempts):
                        logger.info(
                            f"\n  Repair attempt {attempt + 1}/{self.max_repair_attempts}"
                        )

                        # Get repair suggestions
                        repair_ctx = AgentContext(
                            data={
                                "broken_urls": [{**url_info, "error": current_error}],
                                "language": language,
                            }
                        )
                        repair_output = self.repair_agent.run(repair_ctx)

                        if not repair_output.success:
                            logger.warning(
                                f"  Repair agent failed: {repair_output.errors}"
                            )
                            break

                        # Transfer output data to context
                        if repair_output.data:
                            repair_ctx.update(repair_output.data)

                        repair_suggestions = repair_ctx.get("repair_suggestions", {})
                        repair_result = repair_suggestions.get(original_url, {})
                        repair_attempts.append(repair_result)

                        if repair_result.get(
                            "action"
                        ) == "repair" and repair_result.get("suggested_urls"):
                            # Try each suggested URL
                            for suggested_url in repair_result["suggested_urls"]:
                                logger.info(f"  Testing suggested URL: {suggested_url}")

                                # Validate the suggested URL
                                test_ctx = AgentContext(
                                    data={"extracted_urls": [{"url": suggested_url}]}
                                )
                                test_output = self.validator.run(test_ctx)

                                # Transfer output data to context
                                if test_output.data:
                                    test_ctx.update(test_output.data)

                                if test_output.success:
                                    test_results = test_ctx.get(
                                        "validation_results", []
                                    )
                                    if test_results and test_results[0]["accessible"]:
                                        logger.info(
                                            f"  ✓ Found working URL: {suggested_url}"
                                        )
                                        working_url = suggested_url
                                        break
                                    else:
                                        error_info = (
                                            test_results[0] if test_results else {}
                                        )
                                        logger.warning(
                                            f"  ✗ Still inaccessible: {error_info.get('error', 'Unknown')}"
                                        )
                                        current_error = f"Previous attempt: {suggested_url} - {error_info.get('error', 'Unknown')}"

                            # If we found a working URL, break the repair attempt loop
                            if working_url:
                                break
                        else:
                            logger.info(
                                f"  Repair agent suggests action: {repair_result.get('action', 'unknown')}"
                            )
                            break  # Agent says URL can't be repaired

                    # Store repair result
                    repair_results.append(
                        {
                            "original_url": original_url,
                            "repair_attempts": repair_attempts,
                            "total_attempts": len(repair_attempts),
                            "final_result": "repaired" if working_url else "failed",
                        }
                    )

                    # Add to url_changes
                    if working_url:
                        logger.info(
                            f"\n  ✓ Successfully repaired after {len(repair_attempts)} attempt(s)"
                        )
                        url_changes.append(
                            {
                                "original_url": original_url,
                                "action": "replace",
                                "new_url": working_url,
                            }
                        )
                    else:
                        logger.warning(
                            f"\n  ✗ Failed to repair after {len(repair_attempts)} attempt(s), marking for removal"
                        )
                        url_changes.append(
                            {"original_url": original_url, "action": "remove"}
                        )

            # Step 4: Rewrite query
            processed_query = query
            if url_changes:
                logger.info(
                    f"\n[Step 4] Rewriting query ({len(url_changes)} change(s))..."
                )
                rewrite_ctx = AgentContext(
                    data={
                        "query": query,
                        "url_changes": url_changes,
                        "language": language,
                    }
                )
                rewrite_output = self.rewrite_agent.run(rewrite_ctx)

                # Transfer output data to context
                if rewrite_output.data:
                    rewrite_ctx.update(rewrite_output.data)

                if rewrite_output.success:
                    processed_query = rewrite_ctx.get("rewritten_query", query)
                    logger.info("  Query rewritten successfully")
                else:
                    logger.warning(f"  Query rewriting failed: {rewrite_output.errors}")

            # Build complete result
            url_processing_result = {
                "extraction": {"has_urls": True, "urls": extracted_urls},
                "validation": validation_results,
                "repair": repair_results,
                "url_changes": url_changes,
            }

            # Calculate statistics
            urls_repaired = sum(
                1 for r in repair_results if r["final_result"] == "repaired"
            )
            urls_removed = sum(
                1 for r in repair_results if r["final_result"] == "failed"
            )

            logger.info(f"\n✓ URL Processing Complete:")
            logger.info(f"  URLs extracted: {len(extracted_urls)}")
            logger.info(f"  URLs accessible: {accessible_count}")
            logger.info(f"  URLs repaired: {urls_repaired}")
            logger.info(f"  URLs removed: {urls_removed}")
            logger.info(f"  Query rewritten: {len(url_changes) > 0}")

            return AgentOutput(
                success=True,
                data={
                    "url_processing": url_processing_result,
                    "processed_query": processed_query,
                    "url_stats": {
                        "total_extracted": len(extracted_urls),
                        "accessible": accessible_count,
                        "repaired": urls_repaired,
                        "removed": urls_removed,
                    },
                },
                timings={"url_processing": time.time() - start_time},
            )

        except Exception as e:
            logger.error(f"URL processing failed: {e}")
            return AgentOutput(
                success=False,
                errors=[f"URL processing error: {str(e)}"],
                data={"url_processing": {}, "processed_query": query},
            )
