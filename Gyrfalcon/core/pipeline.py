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
Main Pipeline Coordinator for Gyrfalcon v3

This module orchestrates the entire query synthesis pipeline,
coordinating all components to generate queries for agent frameworks
using problem type trees instead of knowledge graphs.

Supports parallel query generation with configurable number of workers.
"""

import copy
import json
import logging
import multiprocessing
import os
import random
import time
from dataclasses import dataclass
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from config.settings import LLM_CONFIG, PIPELINE_CONFIG
from frameworks.framework_manager import FrameworkConfigManager, FrameworkPersona
from llm_interface.query_generator import (
    GeneratedQuery,
    LLMClient,
    QueryExporter,
    QueryGenerationResult,
    QueryGenerator,
)
from problem_type_tree import ProblemTypeTreeManager, TagTrace
from problem_type_tree.visualizer import ProblemTypeTreeVisualizer
from utils.env_loader import load_env_file

logger = logging.getLogger(__name__)


def _load_framework_description(
    framework_dir: Path, framework_name: str, language: str
) -> Optional[str]:
    """
    Load framework description from framework_config.yaml.

    Args:
        framework_dir: Path to frameworks directory
        framework_name: Name of the framework
        language: Language for description ("english" or "chinese")

    Returns:
        Framework description string or None if not found
    """
    try:
        config_path = framework_dir / framework_name / "framework_config.yaml"
        if not config_path.exists():
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config or "framework" not in config:
            return None

        framework_info = config["framework"]
        desc_key = (
            "description_zh" if language.lower() == "chinese" else "description_en"
        )

        return framework_info.get(desc_key)
    except Exception as e:
        logger.warning(
            f"Failed to load framework description for {framework_name}: {e}"
        )
        return None


def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return a copy of metadata with fuzzifier.original_query removed to avoid duplication
    now that the original content is exposed at the top level.
    """
    if not metadata:
        return {}

    sanitized = copy.deepcopy(metadata)
    fuzz_meta = sanitized.get("fuzzifier")
    if isinstance(fuzz_meta, dict):
        fuzz_meta.pop("original_query", None)
    return sanitized


@dataclass
class PipelineRunConfig:
    """Configuration for a single pipeline run"""

    framework_name: str
    num_queries: int = 10
    num_workers: int = 1  # Number of parallel workers for query generation
    language: str = "english"  # Language for query generation: "english" or "chinese"
    export_format: str = "jsonl"  # "jsonl" or "json"
    output_dir: Optional[str] = None
    debug_prompts: bool = False  # Debug mode: save prompts and stop before LLM calls
    generate_visualization: bool = (
        True  # Generate HTML visualization of problem type tree
    )
    difficulty_distribution: Optional[Dict[str, float]] = (
        None  # Custom difficulty distribution (e.g., {'easy': 0.2, 'medium': 0.5, 'hard': 0.3})
    )
    web_search_probability: float = 0.0
    web_search_api_key: Optional[str] = None
    web_search_max_results: int = 5
    web_search_endpoint: Optional[str] = None
    problem_type_expand_probability: float = 0.1
    fuzzify_probability: float = 0.0
    include_framework_description: bool = (
        False  # Include framework description in prompts
    )
    enable_file_analysis: bool = (
        False  # Enable file requirement analysis and file downloading agents
    )
    enable_url_processing: bool = (
        False  # Enable URL processing (extraction, validation, repair)
    )

    def __post_init__(self):
        """Normalize framework name: convert hyphens to underscores for internal consistency"""
        self.framework_name = self.framework_name.replace("-", "_")

        # Set default difficulty distribution if not provided
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {"easy": 0.20, "medium": 0.50, "hard": 0.30}

        self.web_search_probability = max(
            0.0, min(1.0, float(self.web_search_probability or 0.0))
        )
        self.web_search_max_results = max(1, int(self.web_search_max_results or 1))
        self.problem_type_expand_probability = max(
            0.0, min(1.0, float(self.problem_type_expand_probability or 0.1))
        )
        self.fuzzify_probability = max(
            0.0, min(1.0, float(self.fuzzify_probability or 0.0))
        )
        if self.web_search_probability > 0 and not self.web_search_api_key:
            self.web_search_api_key = os.getenv("SERPER_API_KEY")


@dataclass
class PipelineRunResult:
    """Result of a pipeline run"""

    framework_name: str
    total_queries_generated: int
    execution_time: float
    output_files: List[str]
    statistics: Dict[str, Any]
    errors: List[str]
    visualization_file: Optional[str] = None  # Path to HTML visualization file


def _generate_single_query_worker(args):
    """
    Worker function for parallel query generation.
    This function runs in a separate process.

    Args:
        args: Tuple of (worker_id, query_idx, base_dir, framework_name, language,
        debug_prompts, output_file, difficulty_distribution, web_search_probability,
        web_search_api_key, web_search_max_results, web_search_endpoint,
        problem_type_expand_probability, fuzzify_probability, include_framework_description,
        enable_file_analysis)

    Returns:
        Dict with query result and metadata
    """
    (
        worker_id,
        query_idx,
        base_dir,
        framework_name,
        language,
        debug_prompts,
        output_file,
        difficulty_distribution,
        web_search_probability,
        web_search_api_key,
        web_search_max_results,
        web_search_endpoint,
        problem_type_expand_probability,
        fuzzify_probability,
        include_framework_description,
        enable_file_analysis,
        enable_url_processing,
    ) = args

    try:
        load_env_file(Path(base_dir) / ".env")
        # Initialize components in worker process
        frameworks_dir = Path(base_dir) / "frameworks"

        # Initialize LLM client
        llm_client = LLMClient(**LLM_CONFIG)

        # Initialize framework manager
        framework_manager = FrameworkConfigManager(str(frameworks_dir))
        framework_config = framework_manager.get_framework(framework_name)

        if not framework_config:
            return {
                "success": False,
                "error": f"Framework '{framework_name}' not found",
                "worker_id": worker_id,
                "query_idx": query_idx,
            }

        # Initialize problem type tree manager
        problem_type_tree_manager = ProblemTypeTreeManager(
            str(frameworks_dir),
            llm_client=llm_client,
            framework_manager=framework_manager,
        )
        if problem_type_tree_manager.tag_generator:
            problem_type_tree_manager.tag_generator.new_tag_probability = (
                problem_type_expand_probability
            )

        # Initialize query generator
        query_generator = QueryGenerator(llm_client)
        web_search_config = {
            "probability": web_search_probability,
            "api_key": web_search_api_key,
            "max_results": web_search_max_results,
        }
        if web_search_endpoint:
            web_search_config["endpoint"] = web_search_endpoint
        query_generator.set_web_search_config(web_search_config)
        query_generator.set_fuzzifier_probability(fuzzify_probability)
        query_generator.set_file_analysis_enabled(enable_file_analysis)
        query_generator.set_url_processing_enabled(enable_url_processing)

        # Generate query
        round_start = time.time()
        timings = {}

        # Select random persona
        t0 = time.time()
        persona = random.choice(framework_config.personas)
        timings["persona_selection"] = time.time() - t0

        # Generate tag trace
        t0 = time.time()
        tag_trace = problem_type_tree_manager.generate_tag_trace_for_persona(
            framework_name=framework_name, persona=persona.persona, language=language
        )
        timings["tag_trace_generation"] = time.time() - t0

        # Select problem type from trace
        t0 = time.time()
        selected_node, problem_type = (
            problem_type_tree_manager.select_problem_type_from_trace(
                trace=tag_trace, persona=persona.persona, language=language
            )
        )
        timings["problem_type_selection"] = time.time() - t0

        # Load framework description if requested
        framework_description = None
        if include_framework_description:
            framework_description = _load_framework_description(
                frameworks_dir, framework_name, language
            )
            if framework_description:
                logger.info(f"Loaded framework description for {framework_name}")

        # Generate queries
        t0 = time.time()
        result = query_generator.generate_queries(
            persona=persona,
            tag_trace=tag_trace,
            problem_type=problem_type,
            language=language,
            debug_mode=debug_prompts,
            difficulty_distribution=difficulty_distribution,
            framework_name=framework_name,
            framework_description=framework_description,
        )
        timings["query_generation_total"] = time.time() - t0

        # Extract detailed timings
        if "timings" in result.generation_metadata:
            for key, value in result.generation_metadata["timings"].items():
                timings[f"qgen_{key}"] = value

        round_total = time.time() - round_start
        timings["round_total"] = round_total

        # Write to file (JSONL format for parallel safety)
        if result.queries and output_file:
            for query in result.queries:
                query_metadata = query.metadata or {}
                fuzz_meta = query_metadata.get("fuzzifier", {})
                sanitized_metadata = _sanitize_metadata(query_metadata)

                combined_metadata = {
                    "persona": persona.get_persona(language),
                    "problem_type": problem_type,
                    "tag_trace": tag_trace.get_label_string(),
                    "timestamp": time.time(),
                }
                combined_metadata.update(result.generation_metadata or {})
                combined_metadata.update(sanitized_metadata)

                query_dict = {
                    "query": query.content,  # Fixed: use 'content' attribute
                    "difficulty": query.difficulty,
                    "trace_context": tag_trace.get_labels(),  # Fixed: correct method name
                    "framework": framework_name,  # Add framework field
                    "metadata": combined_metadata,
                    "requires_local_files": query_metadata.get(
                        "requires_local_files", False
                    ),
                    "used_web_search": query_metadata.get("used_web_search", False),
                    "fuzzified": bool(fuzz_meta.get("applied")),
                }
                if fuzz_meta.get("applied") and fuzz_meta.get("original_query"):
                    query_dict["original_query"] = fuzz_meta.get("original_query")

                # Append to JSONL file (thread-safe on most systems)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(query_dict, ensure_ascii=False) + "\n")

        return {
            "success": True,
            "worker_id": worker_id,
            "query_idx": query_idx,
            "num_queries": len(result.queries) if result.queries else 0,
            "timings": timings,
            "persona": persona.get_persona(language),
            "problem_type": problem_type,
            "tag_trace": tag_trace.get_label_string(),
        }

    except Exception as e:
        logger.error(f"Worker {worker_id} failed on query {query_idx}: {e}")
        return {
            "success": False,
            "worker_id": worker_id,
            "query_idx": query_idx,
            "error": str(e),
        }


class GyrfalconPipeline:
    """
    Main pipeline coordinator that orchestrates the entire query synthesis process
    using problem type trees.
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.frameworks_dir = self.base_dir / "frameworks"
        self.output_dir = self.base_dir / "output"

        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)

        # Load environment variables from .env if available
        load_env_file(self.base_dir / ".env")

        # Initialize LLM client first (needed by other components)
        self.llm_client = LLMClient(**LLM_CONFIG)

        # Initialize components
        self.framework_manager = FrameworkConfigManager(str(self.frameworks_dir))
        self.problem_type_tree_manager = ProblemTypeTreeManager(
            str(self.frameworks_dir),
            llm_client=self.llm_client,  # Pass LLM client for new tag generation
            framework_manager=self.framework_manager,  # Pass framework manager for framework config access
        )
        self.tree_visualizer = ProblemTypeTreeVisualizer(self.problem_type_tree_manager)
        self.query_generator = QueryGenerator(self.llm_client)
        logger.info(f"Initialized Gyrfalcon v3 Pipeline in {base_dir}")

    def run_pipeline(self, config: PipelineRunConfig) -> PipelineRunResult:
        """
        Run the complete pipeline for a specific framework.
        """
        start_time = time.time()
        errors = []
        output_files = []

        logger.info(f"Starting pipeline run for framework: {config.framework_name}")

        try:
            # Validate framework exists
            framework_config = self.framework_manager.get_framework(
                config.framework_name
            )
            if not framework_config:
                raise ValueError(f"Framework '{config.framework_name}' not found")

            # Load problem type tree for this framework
            try:
                problem_type_tree = self.problem_type_tree_manager.load_framework_tree(
                    config.framework_name
                )
                logger.info(f"Loaded problem type tree for {config.framework_name}")
            except FileNotFoundError as e:
                raise ValueError(
                    f"Problem type tree not found for framework '{config.framework_name}': {e}"
                )

            # Get tree statistics
            tree_stats = problem_type_tree.get_tree_statistics()
            logger.info(f"Tree stats: {tree_stats}")

            # Apply runtime configuration for tag expansion probability
            if self.problem_type_tree_manager.tag_generator:
                self.problem_type_tree_manager.tag_generator.new_tag_probability = (
                    config.problem_type_expand_probability
                )

            # Configure web search integration
            web_search_config = {
                "probability": config.web_search_probability,
                "api_key": config.web_search_api_key or os.getenv("SERPER_API_KEY"),
                "max_results": config.web_search_max_results,
            }
            if config.web_search_endpoint:
                web_search_config["endpoint"] = config.web_search_endpoint
            self.query_generator.set_web_search_config(web_search_config)
            self.query_generator.set_fuzzifier_probability(config.fuzzify_probability)
            self.query_generator.set_file_analysis_enabled(config.enable_file_analysis)
            self.query_generator.set_url_processing_enabled(
                config.enable_url_processing
            )

            # Display configuration summary with status indicators
            if config.web_search_probability > 0:
                print(
                    f"🔍 Web search enabled (probability: {config.web_search_probability:.2f})"
                )
            if config.fuzzify_probability > 0:
                print(
                    f"🔮 Query fuzzification enabled (probability: {config.fuzzify_probability:.2f})"
                )
            if config.problem_type_expand_probability > 0:
                print(
                    f"🌱 Dynamic tag expansion enabled (probability: {config.problem_type_expand_probability:.2f})"
                )
            if config.enable_url_processing:
                print(f"🔗 URL processing enabled (extraction, validation, repair)")
            if (
                config.web_search_probability == 0
                and config.fuzzify_probability == 0
                and config.problem_type_expand_probability == 0
                and not config.enable_url_processing
            ):
                print(
                    "📝 Running with standard query generation (no optional features enabled)"
                )

            # Generate queries
            all_results = []
            total_queries = 0

            # Create output file path once
            output_dir = (
                Path(config.output_dir) if config.output_dir else self.output_dir
            )
            output_dir.mkdir(exist_ok=True)
            timestamp = int(time.time())

            if config.export_format.lower() == "jsonl":
                output_file = (
                    output_dir / f"{config.framework_name}_queries_{timestamp}.jsonl"
                )
            else:
                output_file = (
                    output_dir / f"{config.framework_name}_queries_{timestamp}.json"
                )
                # For JSON, we'll need to collect all and write at end
                json_mode = True

            print(f"\n💾 Output file: {output_file.name}")

            # Decide execution mode: parallel or sequential
            if config.num_workers > 1 and not config.debug_prompts:
                # Parallel execution with multiprocessing
                print(f"🚀 Running with {config.num_workers} parallel workers")
                print(f"   Each query will be appended immediately after generation\n")

                total_queries = self._run_parallel(config, output_file)

            else:
                # Sequential execution (original behavior)
                if config.num_workers > 1 and config.debug_prompts:
                    print(
                        f"⚠️  Debug mode: forcing sequential execution (num_workers={config.num_workers} ignored)"
                    )
                print(f"   Each query will be appended immediately after generation\n")

                total_queries = self._run_sequential(
                    config, output_file, framework_config
                )

            # Output files already created by worker methods
            output_files.append(str(output_file))

            # Save final sampling stats
            if config.framework_name in self.problem_type_tree_manager.sampling_stats:
                sampling_stats = self.problem_type_tree_manager.sampling_stats[
                    config.framework_name
                ]
                sampling_stats.save_stats()
                logger.info(
                    f"Saved final sampling statistics: {sampling_stats.total_samples} total samples"
                )

            # Reload tree to get latest version with any new tags generated during execution
            problem_type_tree = self.problem_type_tree_manager.load_framework_tree(
                config.framework_name
            )
            logger.info(
                f"Reloaded problem type tree to get final statistics (may include newly generated tags)"
            )

            # Generate statistics
            statistics = self._generate_statistics(
                problem_type_tree, all_results, framework_config
            )

            # Save statistics
            stats_file = output_dir / f"{config.framework_name}_stats_{timestamp}.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            output_files.append(str(stats_file))

            # Generate visualization if requested
            visualization_file = None
            if config.generate_visualization:
                try:
                    viz_file = (
                        output_dir
                        / f"{config.framework_name}_tree_visualization_{timestamp}.html"
                    )
                    visualization_file = self.tree_visualizer.generate_html(
                        framework_name=config.framework_name,
                        output_path=str(viz_file),
                        language=config.language,
                    )
                    output_files.append(visualization_file)
                    logger.info(f"Generated tree visualization: {visualization_file}")
                    print(f"\n🎨 Problem Type Tree Visualization:")
                    print(f"   📁 {Path(visualization_file).name}")
                    print(
                        f"   🌐 Open in browser: file://{Path(visualization_file).absolute()}"
                    )
                except Exception as e:
                    error_msg = f"Failed to generate visualization: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            execution_time = time.time() - start_time

            result = PipelineRunResult(
                framework_name=config.framework_name,
                total_queries_generated=total_queries,
                execution_time=execution_time,
                output_files=output_files,
                statistics=statistics,
                errors=errors,
                visualization_file=visualization_file,
            )

            logger.info(
                f"Pipeline run completed for {config.framework_name}: "
                f"{total_queries} queries in {execution_time:.2f}s"
            )

            return result

        except Exception as e:
            error_msg = f"Pipeline run failed for {config.framework_name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

            execution_time = time.time() - start_time
            return PipelineRunResult(
                framework_name=config.framework_name,
                total_queries_generated=0,
                execution_time=execution_time,
                output_files=output_files,
                statistics={},
                errors=errors,
                visualization_file=None,
            )

    def _run_parallel(self, config: PipelineRunConfig, output_file: Path) -> int:
        """
        Run query generation in parallel using multiprocessing.

        Args:
            config: Pipeline configuration
            output_file: Path to output JSONL file

        Returns:
            Total number of queries generated
        """
        # Prepare worker arguments
        worker_args = [
            (
                i % config.num_workers,  # worker_id
                i,  # query_idx
                str(self.base_dir),  # base_dir
                config.framework_name,  # framework_name
                config.language,  # language
                config.debug_prompts,  # debug_prompts
                str(output_file),  # output_file
                config.difficulty_distribution,  # difficulty_distribution
                config.web_search_probability,
                config.web_search_api_key or os.getenv("SERPER_API_KEY"),
                config.web_search_max_results,
                config.web_search_endpoint,
                config.problem_type_expand_probability,
                config.fuzzify_probability,
                config.include_framework_description,  # include_framework_description
                config.enable_file_analysis,  # enable_file_analysis
                config.enable_url_processing,  # enable_url_processing
            )
            for i in range(config.num_queries)
        ]

        # Create process pool and execute
        total_queries = 0
        completed_count = 0
        failed_count = 0

        print(
            f"Starting {config.num_queries} query generation tasks across {config.num_workers} workers...\n"
        )

        with Pool(processes=config.num_workers) as pool:
            # Use imap_unordered for better performance with progress tracking
            for result in pool.imap_unordered(
                _generate_single_query_worker, worker_args
            ):
                if result["success"]:
                    completed_count += 1
                    total_queries += result["num_queries"]

                    # Display progress
                    print(
                        f"✅ Query {completed_count}/{config.num_queries} completed "
                        f"(Worker {result['worker_id']}, {result['num_queries']} variants)"
                    )
                    print(f"   Persona: {result['persona'][:80]}...")
                    print(f"   Problem Type: {result['problem_type']}")
                    print(f"   Time: {result['timings']['round_total']:.2f}s\n")
                else:
                    failed_count += 1
                    print(
                        f"❌ Query {result['query_idx']} failed (Worker {result['worker_id']})"
                    )
                    print(f"   Error: {result.get('error', 'Unknown error')}\n")

        print(f"\n✅ PARALLEL EXECUTION COMPLETED")
        print(f"   📁 File: {output_file.name}")
        print(f"   📊 Total queries: {total_queries}")
        print(f"   ✅ Successful: {completed_count}")
        print(f"   ❌ Failed: {failed_count}")

        return total_queries

    def _run_sequential(
        self, config: PipelineRunConfig, output_file: Path, framework_config
    ) -> int:
        """
        Run query generation sequentially (original behavior).

        Args:
            config: Pipeline configuration
            output_file: Path to output file
            framework_config: Framework configuration

        Returns:
            Total number of queries generated
        """
        all_results = []
        total_queries = 0

        for i in range(config.num_queries):
            try:
                # === TIME TRACKING START ===
                round_start = time.time()
                timings = {}

                # Select random persona
                t0 = time.time()
                persona = random.choice(framework_config.personas)
                timings["persona_selection"] = time.time() - t0

                # Generate tag trace
                t0 = time.time()
                tag_trace = (
                    self.problem_type_tree_manager.generate_tag_trace_for_persona(
                        framework_name=config.framework_name,
                        persona=persona.persona,
                        language=config.language,
                    )
                )
                timings["tag_trace_generation"] = time.time() - t0

                # Select problem type from trace
                t0 = time.time()
                selected_node, problem_type = (
                    self.problem_type_tree_manager.select_problem_type_from_trace(
                        trace=tag_trace,
                        persona=persona.persona,
                        language=config.language,
                    )
                )
                timings["problem_type_selection"] = time.time() - t0

                # Display query generation configuration
                self._display_query_config(
                    i + 1,
                    config.num_queries,
                    persona,
                    tag_trace,
                    problem_type,
                    config.language,
                )

                # Load framework description if requested
                framework_description = None
                if config.include_framework_description:
                    framework_description = _load_framework_description(
                        self.frameworks_dir, config.framework_name, config.language
                    )
                    if framework_description:
                        logger.info(
                            f"Loaded framework description for {config.framework_name}"
                        )

                # Generate queries
                t0 = time.time()
                result = self.query_generator.generate_queries(
                    persona=persona,
                    tag_trace=tag_trace,
                    problem_type=problem_type,
                    language=config.language,
                    debug_mode=config.debug_prompts,
                    difficulty_distribution=config.difficulty_distribution,
                    framework_name=config.framework_name,
                    framework_description=framework_description,
                )
                timings["query_generation_total"] = time.time() - t0

                # Extract detailed timings
                if "timings" in result.generation_metadata:
                    for key, value in result.generation_metadata["timings"].items():
                        timings[f"qgen_{key}"] = value

                # Check if this is debug mode and stop execution
                if config.debug_prompts and result.generation_metadata.get(
                    "debug_mode"
                ):
                    prompt_file = result.generation_metadata.get("prompt_file")
                    logger.info(f"🐛 DEBUG MODE COMPLETE")
                    logger.info(f"📝 Prompt saved to: {prompt_file}")
                    logger.info(
                        f"🔍 Please review the prompt file and run without --debug-prompts to continue"
                    )
                    return 0  # Return 0 for debug mode

                # Append query to file immediately (JSONL only)
                t0 = time.time()
                if result.queries:
                    if config.export_format.lower() == "jsonl":
                        for query in result.queries:
                            total_queries += 1
                            self._append_query_to_file(
                                output_file,
                                query,
                                result,
                                total_queries,
                                config.framework_name,
                            )
                    else:
                        # JSON format: just count, will save at end
                        total_queries += len(result.queries)
                timings["file_writing"] = time.time() - t0

                # Calculate total round time
                round_total = time.time() - round_start
                timings["round_total"] = round_total

                # Store result for statistics
                all_results.append(result)

                logger.info(
                    f"Generation {i+1}/{config.num_queries}: {len(result.queries)} queries"
                )

                # Display detailed timing breakdown
                print(f"\n⏱️  TIMING BREAKDOWN (Round {i+1}):")
                print(f"   {'Module':<30} {'Time (s)':<10} {'%':<8}")
                print(f"   {'-'*50}")
                for module, duration in sorted(
                    timings.items(), key=lambda x: x[1], reverse=True
                ):
                    if module != "round_total":
                        pct = (duration / round_total * 100) if round_total > 0 else 0
                        print(f"   {module:<30} {duration:>8.3f}s   {pct:>5.1f}%")
                print(f"   {'-'*50}")
                print(f"   {'TOTAL':<30} {round_total:>8.3f}s   100.0%\n")

            except Exception as e:
                error_msg = f"Error in generation round {i+1}: {e}"
                logger.error(error_msg)

        # For JSON format, save all results at end
        if config.export_format.lower() == "json":
            print(f"\n📦 FINAL JSON SAVE")
            print(f"   🔄 Combining all {total_queries} queries into JSON format...")
            QueryExporter.export_to_json(
                all_results, output_file, framework_name=config.framework_name
            )
            print(f"   📁 File: {output_file.name}")
            print(f"   📊 Total queries: {total_queries}")
        else:
            print(f"\n✅ ALL QUERIES SAVED")
            print(f"   📁 File: {output_file.name}")
            print(f"   📊 Total queries: {total_queries}")
            print(f"   💡 Each query was appended immediately after generation")

        return total_queries

    def _generate_statistics(
        self, problem_type_tree, results: List[QueryGenerationResult], framework_config
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for the pipeline run.
        """
        # Problem type tree statistics
        tree_stats = problem_type_tree.get_tree_statistics()

        # Query statistics
        query_difficulties = []
        problem_types_used = []
        fuzzified_count = 0
        web_search_count = 0
        local_files_count = 0

        for result in results:
            for query in result.queries:
                query_difficulties.append(query.difficulty)

                # Check fuzzification status
                if query.metadata:
                    fuzz_meta = query.metadata.get("fuzzifier", {})
                    if fuzz_meta.get("applied"):
                        fuzzified_count += 1

                    # Check other metadata
                    if query.metadata.get("used_web_search"):
                        web_search_count += 1
                    if query.metadata.get("requires_local_files"):
                        local_files_count += 1

            if result.problem_type:
                problem_types_used.append(result.problem_type)

        difficulty_counts = {}
        for diff in query_difficulties:
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        problem_type_counts = {}
        for pt in problem_types_used:
            problem_type_counts[pt] = problem_type_counts.get(pt, 0) + 1

        statistics = {
            "problem_type_tree": tree_stats,
            "queries": {
                "total_generated": len(query_difficulties),
                "difficulty_distribution": difficulty_counts,
                "average_per_generation": (
                    len(query_difficulties) / len(results) if results else 0
                ),
                "fuzzified_queries": fuzzified_count,
                "web_search_queries": web_search_count,
                "local_files_queries": local_files_count,
            },
            "problem_types": {
                "unique_types_used": len(problem_type_counts),
                "type_distribution": problem_type_counts,
            },
            "framework": {
                "total_personas": len(framework_config.personas),
                "total_subagents": len(framework_config.subagents),
                "total_tools": len(framework_config.tools),
            },
            "generation_metadata": {
                "total_generation_rounds": len(results),
                "successful_rounds": len([r for r in results if r.queries]),
                "failed_rounds": len([r for r in results if not r.queries]),
            },
        }

        return statistics

    def run_batch_pipeline(
        self, configs: List[PipelineRunConfig]
    ) -> List[PipelineRunResult]:
        """
        Run pipeline for multiple frameworks in batch.
        """
        results = []

        for config in configs:
            try:
                result = self.run_pipeline(config)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch pipeline failed for {config.framework_name}: {e}")
                results.append(
                    PipelineRunResult(
                        framework_name=config.framework_name,
                        total_queries_generated=0,
                        execution_time=0,
                        output_files=[],
                        statistics={},
                        errors=[str(e)],
                    )
                )

        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current status of the pipeline.
        """
        framework_summary = self.framework_manager.get_summary()

        tree_status = {}
        for fw_name in framework_summary.get("frameworks", []):
            try:
                tree = self.problem_type_tree_manager.get_tree(fw_name)
                if tree:
                    tree_status[fw_name] = tree.get_tree_statistics()
            except:
                pass

        return {
            "frameworks": framework_summary,
            "problem_type_trees": tree_status,
            "output_directory": str(self.output_dir),
        }

    def _display_query_config(
        self,
        current_round: int,
        total_rounds: int,
        persona: FrameworkPersona,
        trace: TagTrace,
        problem_type: str,
        language: str = "english",
    ):
        """
        Display the configuration for the current query generation round.
        """
        # Create a visual separator
        print("\n" + "=" * 80)
        print(f"🎯 QUERY GENERATION CONFIG - Round {current_round}/{total_rounds}")
        print("=" * 80)

        # Language Information
        lang_emoji = "🇺🇸" if language.lower() == "english" else "🇨🇳"
        lang_display = "English" if language.lower() == "english" else "中文"
        print(f"🌐 LANGUAGE: {lang_emoji} {lang_display}")

        # Persona Information
        print(f"\n👤 SELECTED PERSONA:")
        print(f"   {persona.get_persona(language)}")

        # Tag Trace Information
        print(f"\n🔗 PROBLEM TYPE TRACE ({len(trace.nodes)} levels):")
        trace_path = trace.get_label_string()
        print(f"   {trace_path}")

        # Selected Problem Type
        print(f"\n🎨 SELECTED PROBLEM TYPE:")
        print(f"   {problem_type}")

        print("=" * 80)
        print("🚀 Generating queries...")
        print()

    def _append_query_to_file(
        self,
        output_file: Path,
        query: "GeneratedQuery",
        result: QueryGenerationResult,
        query_number: int,
        framework_name: str = None,
    ):
        """
        Append a single query to the output file immediately after generation.

        Args:
            output_file: Path to the output JSONL file
            query: The GeneratedQuery object to save
            result: The QueryGenerationResult containing metadata
            query_number: Sequential query number (1, 2, 3, ...)
            framework_name: Framework name to include in export (optional)
        """
        try:
            # Prepare single query data
            query_metadata = query.metadata or {}
            fuzz_meta = query_metadata.get("fuzzifier", {})
            sanitized_metadata = _sanitize_metadata(query_metadata)

            query_data = {
                "query": query.content,
                "difficulty": query.difficulty,
                "trace_context": result.trace_context,
                "framework": framework_name,  # Add framework field
                "problem_type": result.problem_type,
                "requires_local_files": query_metadata.get(
                    "requires_local_files", False
                ),
                "used_web_search": query_metadata.get("used_web_search", False),
                "fuzzified": bool(fuzz_meta.get("applied")),
                "metadata": {
                    **sanitized_metadata,
                    **(result.generation_metadata or {}),
                },
            }
            if fuzz_meta.get("applied") and fuzz_meta.get("original_query"):
                query_data["original_query"] = fuzz_meta.get("original_query")

            # Append to JSONL file
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(query_data, ensure_ascii=False) + "\n")

            # Print save confirmation
            difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(
                query.difficulty, "⚪"
            )

            # Add fuzzification indicator
            fuzz_meta = query.metadata.get("fuzzifier", {}) if query.metadata else {}
            fuzz_indicator = "🔮" if fuzz_meta.get("applied") else ""

            # Add web search indicator
            web_indicator = "🔍" if query.metadata.get("used_web_search") else ""

            # Add file requirement indicator
            file_indicator = "📁" if query.metadata.get("requires_local_files") else ""

            indicators = f"{fuzz_indicator}{web_indicator}{file_indicator}".strip()
            indicator_str = f" {indicators}" if indicators else ""

            print(
                f"💾 Query #{query_number:04d} appended: {difficulty_emoji} {query.difficulty.upper()}{indicator_str}"
            )
            print(
                f"   📝 {query.content[:80]}{'...' if len(query.content) > 80 else ''}"
            )
            print()

            logger.info(f"Appended query #{query_number} to {output_file.name}")

        except Exception as e:
            logger.error(f"Failed to append query #{query_number}: {e}")
            print(f"❌ APPEND FAILED - Query #{query_number}: {e}")

    def _save_single_query(
        self,
        config: PipelineRunConfig,
        query: "GeneratedQuery",
        result: QueryGenerationResult,
        query_number: int,
    ):
        """
        DEPRECATED: This method is no longer used.
        Queries are now appended immediately via _append_query_to_file().
        """
        pass

    def _save_immediate_results(
        self,
        config: PipelineRunConfig,
        round_results: List[QueryGenerationResult],
        round_number: int,
        total_queries: int,
    ):
        """
        DEPRECATED: This method is no longer used.
        Individual queries are now saved immediately via _save_single_query().
        """
        pass


def create_pipeline_config(framework_name: str, **kwargs) -> PipelineRunConfig:
    """
    Convenience function to create a pipeline run configuration.
    """
    return PipelineRunConfig(framework_name=framework_name, **kwargs)
