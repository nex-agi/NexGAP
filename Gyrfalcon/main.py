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
Main entry point for Gyrfalcon v3 Pipeline

Command-line interface for running the query synthesis pipeline.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# IMPORTANT: Load .env files BEFORE importing config.settings
# This ensures environment variables are available when settings.py is loaded
from utils.env_loader import load_env_file

gyrfalcon_dir = Path(__file__).parent
parent_dir = gyrfalcon_dir.parent
load_env_file(parent_dir / ".env")  # NexGAP root
load_env_file(gyrfalcon_dir / ".env")  # Gyrfalcon directory
load_env_file(Path.cwd() / ".env")  # Current directory

# Now safe to import modules that use settings
from core.pipeline import GyrfalconPipeline, PipelineRunConfig


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("gyrfalcon_pipeline.log"),
        ],
    )


def parse_difficulty_distribution(distribution_str: str) -> dict:
    """
    Parse difficulty distribution string into a dictionary.

    Args:
        distribution_str: String in format "easy:0.2,medium:0.5,hard:0.3"

    Returns:
        Dictionary with difficulty levels as keys and probabilities as values

    Raises:
        ValueError: If format is invalid or probabilities don't sum to 1.0
    """
    try:
        parts = distribution_str.split(",")
        distribution = {}

        for part in parts:
            if ":" not in part:
                raise ValueError(
                    f"Invalid format for part '{part}'. Expected format: 'difficulty:probability'"
                )

            difficulty, prob_str = part.strip().split(":")
            difficulty = difficulty.strip().lower()

            if difficulty not in ["easy", "medium", "hard"]:
                raise ValueError(
                    f"Invalid difficulty level '{difficulty}'. Must be one of: easy, medium, hard"
                )

            probability = float(prob_str.strip())

            if probability < 0 or probability > 1:
                raise ValueError(
                    f"Probability for '{difficulty}' must be between 0 and 1, got {probability}"
                )

            distribution[difficulty] = probability

        # Validate that all difficulties are present
        required_difficulties = {"easy", "medium", "hard"}
        if set(distribution.keys()) != required_difficulties:
            missing = required_difficulties - set(distribution.keys())
            extra = set(distribution.keys()) - required_difficulties
            msg = []
            if missing:
                msg.append(f"Missing difficulties: {missing}")
            if extra:
                msg.append(f"Extra difficulties: {extra}")
            raise ValueError(". ".join(msg))

        # Validate that probabilities sum to 1.0 (with small tolerance for floating point)
        total = sum(distribution.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Probabilities must sum to 1.0, got {total:.4f}")

        return distribution

    except ValueError as e:
        raise ValueError(
            f"Error parsing difficulty distribution '{distribution_str}': {e}"
        )
    except Exception as e:
        raise ValueError(
            f"Unexpected error parsing difficulty distribution '{distribution_str}': {e}"
        )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Gyrfalcon v3 Query Synthesis Pipeline"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for the pipeline (default: current directory)",
    )
    parser.add_argument(
        "--framework",
        type=str,
        required=True,
        help="Name of the agent framework to generate queries for",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of query generation rounds (default: 10)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for query generation (default: 1, max recommended: CPU cores)",
    )
    parser.add_argument(
        "--language",
        choices=["english", "chinese"],
        default="english",
        help="Language for query generation (default: english)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: base-dir/output)",
    )
    parser.add_argument(
        "--export-format",
        choices=["json", "jsonl"],
        default="jsonl",
        help="Export format for results (default: jsonl)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--batch-config", type=str, help="Path to JSON file with batch configuration"
    )
    parser.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Debug mode: save prompts to files and stop before LLM calls",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip generating HTML tree visualization (default: generate)",
    )
    parser.add_argument(
        "--difficulty-distribution",
        type=str,
        default="easy:0.2,medium:0.5,hard:0.3",
        help="Difficulty distribution for query selection (format: easy:0.2,medium:0.5,hard:0.3, must sum to 1.0, default: easy:0.2,medium:0.5,hard:0.3)",
    )
    parser.add_argument(
        "--websearch-prob",
        type=float,
        default=0.0,
        help="Probability [0,1] of performing web search enrichment per query (default: 0.0)",
    )
    parser.add_argument(
        "--websearch-api-key",
        type=str,
        default=None,
        help="API key for web search (default: SERPER_API_KEY environment variable)",
    )
    parser.add_argument(
        "--websearch-max-results",
        type=int,
        default=5,
        help="Maximum number of web search results to include (default: 5)",
    )
    parser.add_argument(
        "--websearch-endpoint",
        type=str,
        default=None,
        help="Override the web search endpoint URL (default: Serper.dev)",
    )
    parser.add_argument(
        "--problem-type-expand-prob",
        type=float,
        default=0.1,
        help="Probability [0,1] of expanding the problem type tree with new nodes (default: 0.1)",
    )
    parser.add_argument(
        "--fuzzify-prob",
        type=float,
        default=0.0,
        help="Probability [0,1] of invoking the post-generation fuzzifier subagent (default: 0.0)",
    )
    parser.add_argument(
        "--include-framework-description",
        action="store_true",
        help="Include framework description from framework_config.yaml in query generation prompts (default: False)",
    )
    parser.add_argument(
        "--enable-file-analysis",
        action="store_true",
        help="Enable file requirement analysis and file downloading agents (default: False)",
    )
    parser.add_argument(
        "--enable-url-processing",
        action="store_true",
        help="Enable URL processing (extraction, validation, and repair) for queries (default: False)",
    )

    args = parser.parse_args()

    # Load additional .env from base_dir if specified
    if args.base_dir != ".":
        base_dir_path = Path(args.base_dir).resolve()
        load_env_file(base_dir_path / ".env")

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Parse difficulty distribution
        try:
            difficulty_distribution = parse_difficulty_distribution(
                args.difficulty_distribution
            )
            logger.info(f"Using difficulty distribution: {difficulty_distribution}")
        except ValueError as e:
            logger.error(f"Invalid difficulty distribution: {e}")
            sys.exit(1)

        if not 0.0 <= args.websearch_prob <= 1.0:
            logger.error("--websearch-prob must be between 0 and 1")
            sys.exit(1)
        if not 0.0 <= args.problem_type_expand_prob <= 1.0:
            logger.error("--problem-type-expand-prob must be between 0 and 1")
            sys.exit(1)
        if not 0.0 <= args.fuzzify_prob <= 1.0:
            logger.error("--fuzzify-prob must be between 0 and 1")
            sys.exit(1)
        if args.websearch_max_results <= 0:
            logger.error("--websearch-max-results must be greater than 0")
            sys.exit(1)

        websearch_api_key = args.websearch_api_key or os.getenv("SERPER_API_KEY")

        # Initialize pipeline
        pipeline = GyrfalconPipeline(args.base_dir)

        if args.batch_config:
            # Run batch pipeline
            with open(args.batch_config, "r") as f:
                batch_configs = json.load(f)

            configs = [PipelineRunConfig(**config) for config in batch_configs]
            for cfg in configs:
                if cfg.web_search_probability > 0 and not cfg.web_search_api_key:
                    cfg.web_search_api_key = os.getenv("SERPER_API_KEY")
            results = pipeline.run_batch_pipeline(configs)

            # Print batch results summary
            for result in results:
                print(f"\nFramework: {result.framework_name}")
                print(f"Queries generated: {result.total_queries_generated}")
                print(f"Execution time: {result.execution_time:.2f}s")
                if result.errors:
                    print(f"Errors: {len(result.errors)}")
                print(f"Output files: {result.output_files}")

        else:
            # Run single framework pipeline
            config = PipelineRunConfig(
                framework_name=args.framework,
                num_queries=args.num_queries,
                num_workers=args.num_workers,
                language=args.language,
                export_format=args.export_format,
                output_dir=args.output_dir,
                debug_prompts=args.debug_prompts,
                generate_visualization=not args.no_visualization,
                difficulty_distribution=difficulty_distribution,
                web_search_probability=args.websearch_prob,
                web_search_api_key=websearch_api_key,
                web_search_max_results=args.websearch_max_results,
                web_search_endpoint=args.websearch_endpoint,
                problem_type_expand_probability=args.problem_type_expand_prob,
                fuzzify_probability=args.fuzzify_prob,
                include_framework_description=args.include_framework_description,
                enable_file_analysis=args.enable_file_analysis,
                enable_url_processing=args.enable_url_processing,
            )

            result = pipeline.run_pipeline(config)

            # Print results
            if config.debug_prompts and result.statistics.get("debug_mode"):
                print(f"\n🐛 DEBUG MODE COMPLETE")
                print(f"📝 Prompt saved to: {result.statistics.get('prompt_file')}")
                print(
                    f"🔍 Please review the prompt file and run without --debug-prompts to continue"
                )
                print(f"⏱️  Debug execution time: {result.execution_time:.2f}s")
            else:
                print(f"\nPipeline completed for framework: {result.framework_name}")
                print(f"Queries generated: {result.total_queries_generated}")
                print(f"Execution time: {result.execution_time:.2f}s")

                # Display query feature usage with status indicators
                query_stats = result.statistics.get("queries", {})
                if query_stats:
                    fuzzified = query_stats.get("fuzzified_queries", 0)
                    web_search = query_stats.get("web_search_queries", 0)
                    local_files = query_stats.get("local_files_queries", 0)

                    if fuzzified > 0:
                        print(
                            f"🔮 Fuzzified queries: {fuzzified}/{result.total_queries_generated} ({fuzzified/result.total_queries_generated*100:.1f}%)"
                        )
                    if web_search > 0:
                        print(
                            f"🔍 Web search queries: {web_search}/{result.total_queries_generated} ({web_search/result.total_queries_generated*100:.1f}%)"
                        )
                    if local_files > 0:
                        print(
                            f"📁 Local files queries: {local_files}/{result.total_queries_generated} ({local_files/result.total_queries_generated*100:.1f}%)"
                        )

            if result.errors:
                print(f"\nErrors encountered: {len(result.errors)}")
                for error in result.errors:
                    print(f"  - {error}")

            print(f"\nOutput files:")
            for file_path in result.output_files:
                print(f"  - {file_path}")

            # Print visualization file separately if generated
            if result.visualization_file:
                print(f"\n🎨 Problem Type Tree Visualization:")
                print(f"  📁 {Path(result.visualization_file).name}")
                print(
                    f"  🌐 Open in browser: file://{Path(result.visualization_file).absolute()}"
                )

            # Print problem type tree statistics
            tree_stats = result.statistics.get("problem_type_tree", {})
            if tree_stats:
                print(f"\n🌳 Problem Type Tree Statistics:")
                print(f"  Total nodes: {tree_stats.get('total_nodes', 0)}")
                print(f"  Total paths: {tree_stats.get('total_paths', 0)}")
                print(f"  Max depth: {tree_stats.get('max_depth', 0)}")
                print(f"  Avg depth: {tree_stats.get('avg_depth', 0):.2f}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
