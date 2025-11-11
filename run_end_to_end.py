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
NexGAP End-to-End Script
Complete workflow: query_example.jsonl -> agent execution -> retrieve langfuse trace (via trace_id) -> format conversion

Usage examples:
# Basic usage
uv run run_end_to_end.py --query-filepath data/query_example.jsonl

# Specify framework and count
uv run run_end_to_end.py --query-filepath data/query_example.jsonl --output-dir ./my_output --frameworks deer-flow --max-queries 5

# Parallel processing of multiple queries (background execution, logs saved separately)
uv run run_end_to_end.py --query-filepath data/query_example.jsonl --max-workers 10

# Only run queries, do not fetch traces
uv run run_end_to_end.py --query-filepath data/query_example.jsonl --only-run-query

Background execution notes:
- All queries execute silently in the background, console only shows progress bar and summary
- Each query's detailed log is saved separately to output/logs/query_XXX_framework_timestamp.log
- Supports high concurrency (--max-workers), log files do not interfere with each other
- Can view logs in real-time with: tail -f output/logs/*.log

Before use, you need to properly configure .env, especially LANGFUSE, to record traces
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from converter.trace.get_trace import get_langfuse_trace_by_id
from tqdm import tqdm


class NexGAPRunner:
    def __init__(self, query_file: str, output_dir: str = None):
        self.project_root = Path(__file__).parent
        self._lock = threading.Lock()
        self.nex_agent_path = self.project_root / "NexA4A"
        self.converter_path = self.project_root / "converter"

        # Set paths
        self.query_file = (
            Path(query_file) if not Path(query_file).is_absolute() else Path(query_file)
        )
        if not self.query_file.is_absolute():
            self.query_file = self.project_root / query_file

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
            self.use_framework_subdirs = False
        else:
            self.output_dir = self.project_root / "output"
            self.use_framework_subdirs = True

        # Create output directories
        self.langfuse_trace_dir = self.output_dir / "langfuse_trace"
        self.converted_trace_dir = self.output_dir / "converted_trace"
        self.log_dir = self.output_dir / "logs"
        self._setup_output_directories()

        # Verify paths
        for path, name in [
            (self.nex_agent_path, "NexA4A"),
            (self.converter_path, "Converter"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{name} path not found: {path}")

        if not self.query_file.exists():
            print(f"⚠️ Query file not found: {self.query_file}")
            print(
                "   Please ensure the query file exists or specify with --query-filepath"
            )

        # Print configuration
        for label, path in [
            ("Project root", self.project_root),
            ("nex-agent path", self.nex_agent_path),
            ("Converter path", self.converter_path),
            ("Query file", self.query_file),
            ("Output directory", self.output_dir),
        ]:
            print(f"✅ {label}: {path}")

    def _setup_output_directories(self):
        """Create output directory structure"""
        try:
            for dir_path in [
                self.output_dir,
                self.langfuse_trace_dir,
                self.converted_trace_dir,
                self.log_dir,
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created output directories: {self.output_dir}")
        except Exception as e:
            print(f"⚠️ Failed to create output directories: {e}")
            raise

    def _make_error_result(
        self,
        query_index: int,
        query_data: Dict,
        reason: str,
        trace_id: str = None,
        stop: str = "",
    ):
        """Helper method to create error result"""
        return {
            "success": False,
            "query_index": query_index,
            "query_data": query_data,
            "reason": reason,
            "trace_id": trace_id,
            "stop": stop,
        }

    def get_available_frameworks(self) -> List[str]:
        """Get list of available frameworks"""
        frameworks_dir = self.project_root / "NexA4A" / "src" / "created_subagents"
        if not frameworks_dir.exists():
            return []

        frameworks = []
        for item in frameworks_dir.iterdir():
            if item.is_dir() and (item / "meta.yaml").exists():
                frameworks.append(item.name)

        return sorted(frameworks)

    def load_queries(self, target_frameworks: List[str] = None) -> List[Dict[str, str]]:
        """Load query data, can be filtered by framework"""
        queries = []

        with open(self.query_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    query_data = json.loads(line.strip())

                    # Filter if framework list is specified
                    if target_frameworks:
                        framework = query_data.get("framework", "deer-flow")
                        if framework not in target_frameworks:
                            continue

                    queries.append(query_data)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping line {line_num}: Invalid JSON - {e}")

        if target_frameworks:
            print(
                f"📖 Loaded {len(queries)} queries from {self.query_file.name} for frameworks: {target_frameworks}"
            )
        else:
            print(f"📖 Loaded {len(queries)} queries from {self.query_file.name}")
        return queries

    def run_agent_query(
        self, query: str, framework: str, log_file=None
    ) -> Dict[str, Any]:
        """Run a single query

        Args:
            log_file: Log file object, if provided all output will be written to this file
        """

        def log(msg):
            """Helper function: write to log"""
            if log_file:
                log_file.write(msg + "\n")
                log_file.flush()
            else:
                print(msg)

        log(f"🚀 Running query with {framework} framework:")
        log(f"   Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        result_info = {"success": False, "error": None, "trace_id": None}
        process = None

        try:
            # Run agent query - use uv run to call agent4agent
            # Use cwd parameter instead of os.chdir to avoid multithreading race conditions
            cmd = [
                "uv",
                "run",
                "agent4agent.py",
                "use",
                "--agent",
                framework,
                "--query",
                query,
                "--single-turn",
            ]

            timeout = int(os.getenv("AGENT_EXECUTION_TIMEOUT") or 3600)

            # Let subprocess write directly to log file (if provided) or inherit current stdout/stderr
            stdout_dest = log_file if log_file else None
            stderr_dest = log_file if log_file else None

            process = subprocess.Popen(
                cmd,
                stdout=stdout_dest,
                stderr=stderr_dest,
                text=True,
                cwd=str(self.nex_agent_path),  # Use cwd parameter instead of os.chdir
                start_new_session=True,
            )

            try:
                # Wait for process to complete with timeout
                process.wait(timeout=timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                log(f"❌ Agent query timeout after {timeout} seconds")
                process.kill()
                process.wait()
                result_info["success"] = False
                result_info["error"] = f"Timeout after {timeout} seconds"
                return result_info

            # Check return code
            if returncode == 0:
                log("✅ Agent query completed successfully")

                # Extract trace_id from log file
                if log_file:
                    try:
                        # Flush buffer
                        log_file.flush()
                        # Get current position
                        current_pos = log_file.tell()
                        # Move to start position
                        log_file.seek(0)
                        # Read all content
                        log_content = log_file.read()
                        # Restore position
                        log_file.seek(current_pos)

                        # Extract trace_id
                        import re

                        trace_id_match = re.search(
                            r"LangfuseTraceID:\s*([a-zA-Z0-9-]+)", log_content
                        )
                        if trace_id_match:
                            result_info["trace_id"] = trace_id_match.group(1)
                            log(f"🔍 Extracted trace_id: {result_info['trace_id']}")
                        else:
                            log("⚠️ Could not extract trace_id from log")
                    except Exception as e:
                        log(f"⚠️ Error extracting trace_id: {e}")

                result_info["success"] = True
                result_info["output"] = "(output written to log file)"
            else:
                log(f"❌ Agent query failed with return code: {returncode}")
                result_info["success"] = False
                result_info["error"] = f"Process exited with code {returncode}"

        except Exception as e:
            log(f"❌ Error running agent query: {e}")
            result_info["error"] = str(e)
            # Ensure subprocess cleanup even under other exceptions
            if process and process.poll() is None:
                log("... Exception triggered, killing process ...")
                try:
                    process.kill()
                    process.wait()
                except Exception:
                    pass  # Best effort

        finally:
            # Final safety check
            if process and process.poll() is None:
                print("⚠️ Process still alive in finally block, forcing kill...")
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass

        return result_info

    def get_traces_by_id(self, trace_id: str, query: str, framework: str) -> str:
        """Retrieve Langfuse traces by trace_id"""
        if not trace_id:
            print("❌ No trace_id provided")
            return None

        print(f"📊 Getting traces from Langfuse for trace_id: {trace_id}")

        try:
            # Directly call get_langfuse_trace_by_id function
            langfuse_trace_filepath = get_langfuse_trace_by_id(trace_id)

            if langfuse_trace_filepath and os.path.exists(langfuse_trace_filepath):
                print(f"✅ Traces retrieved successfully: {langfuse_trace_filepath}")

                # Move file to specified output directory and enhance content
                enhanced_filepath = self._enhance_langfuse_trace_file(
                    langfuse_trace_filepath, trace_id, query, framework
                )
                return enhanced_filepath
            else:
                print("❌ Failed to get traces or file not found")
                return None

        except Exception as e:
            print(f"❌ Error calling get_langfuse_trace_by_id: {e}")
            return None

    def _enhance_langfuse_trace_file(
        self, original_filepath: str, trace_id: str, query: str, framework: str
    ) -> str:
        """Enhance langfuse trace file by adding query, framework, langfuse_trace_id, and stop fields"""
        try:
            # Build output directory based on whether to use framework subdirectories
            if self.use_framework_subdirs:
                framework_output_dir = self.output_dir / framework / "langfuse_trace"
            else:
                framework_output_dir = self.output_dir / "langfuse_trace"
            framework_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate new filename (use complete trace_id + microsecond timestamp to ensure uniqueness)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Add microseconds
            new_filename = f"{framework}_{trace_id}_{timestamp}.jsonl"
            new_filepath = framework_output_dir / new_filename

            print(f"📝 Enhancing langfuse trace file: {new_filepath}")

            # Read original file and enhance each line
            with open(original_filepath, "r", encoding="utf-8") as infile:
                with open(new_filepath, "w", encoding="utf-8") as outfile:
                    for line in infile:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                # Add extra fields
                                data["query"] = query
                                data["framework"] = framework
                                data["langfuse_trace_id"] = trace_id
                                data["stop"] = (
                                    ""  # Empty string indicates completely normal execution completion
                                )

                                # Write enhanced data
                                outfile.write(
                                    json.dumps(data, ensure_ascii=False) + "\n"
                                )
                            except json.JSONDecodeError as e:
                                print(f"⚠️ Skipping invalid JSON line: {e}")
                                continue

            print(f"✅ Enhanced langfuse trace saved: {new_filepath}")

            # Delete original file (optional)
            try:
                os.remove(original_filepath)
            except:
                pass

            return str(new_filepath)

        except Exception as e:
            print(f"❌ Error enhancing langfuse trace file: {e}")
            return original_filepath

    def _find_framework_config(self, framework: str) -> Path:
        """Find framework configuration file"""
        for base_dir in ["created_subagents", "created_frameworks"]:
            config_path = (
                self.project_root
                / "NexA4A"
                / "src"
                / base_dir
                / framework
                / "framework_config.yaml"
            )
            if config_path.exists():
                return config_path
        return None

    def convert_traces(
        self, spans_file: str, framework: str, query: str, trace_id: str
    ) -> str:
        """Convert traces to ChatCompletion format"""
        print(f"🔄 Converting traces to ChatCompletion format...")

        if not spans_file or not os.path.exists(spans_file):
            print(f"❌ Spans file not found: {spans_file}")
            return None

        framework_config = self._find_framework_config(framework)
        if not framework_config:
            print(f"❌ Framework config not found for: {framework}")
            return None

        try:
            # Choose converter based on command line flag
            # Default: NexAU XML format (convert_spans_to_chatcompletion_nexau.py)
            # --use-openai-format: OpenAI tool call format (convert_spans_to_chatcompletion.py)
            use_openai = self.use_openai_format
            converter_filename = (
                "convert_spans_to_chatcompletion.py"
                if use_openai
                else "convert_spans_to_chatcompletion_nexau.py"
            )

            converter_module_path = framework_config.parent / converter_filename
            default_converter_path = self.converter_path / "trace" / converter_filename

            if converter_module_path.exists():
                module_path = converter_module_path
                print(f"🛠  Using framework-specific converter: {module_path}")
            else:
                module_path = default_converter_path
                format_type = "OpenAI tool calls" if use_openai else "NexAU XML"
                print(f"🛠  Using default converter ({format_type}): {module_path}")

            if not module_path.exists():
                print(f"❌ Converter module not found: {module_path}")
                return None

            module_name_suffix = (
                framework.replace(" ", "_").replace("-", "_").replace("/", "_")
            )
            module_name = (
                f"convert_spans_to_chatcompletion_{module_name_suffix or 'default'}"
            )

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec or not spec.loader:
                print(f"❌ Failed to load converter module from {module_path}")
                return None

            # Force reload: clear cache BEFORE creating new module instance
            # Also clear related modules that might be cached
            modules_to_clear = [
                module_name,
                "converter.trace.convert_spans_to_chatcompletion",
                "converter.schema.framework_config_schema",
            ]
            for mod in modules_to_clear:
                sys.modules.pop(mod, None)

            converter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(converter_module)

            SpansToChatCompletionConverter = getattr(
                converter_module, "SpansToChatCompletionConverter", None
            )
            if SpansToChatCompletionConverter is None:
                print(f"❌ Converter class not found in module: {module_path}")
                return None

            # Create converter instance
            print(f"🔧 Creating converter instance with config: {framework_config}")
            converter = SpansToChatCompletionConverter(str(framework_config))

            # Generate temporary output filename (output directly to converted_trace directory using final filename)
            spans_path = Path(spans_file)
            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )  # Include microseconds to ensure uniqueness
            final_filename = f"{framework}_{trace_id}_{timestamp}_chatcompletion.jsonl"

            # Determine output path based on whether to use framework subdirectories
            if self.use_framework_subdirs:
                final_output_dir = self.output_dir / framework / "converted_trace"
            else:
                final_output_dir = self.converted_trace_dir
            final_output_dir.mkdir(parents=True, exist_ok=True)

            temp_output_file = str(final_output_dir / final_filename)
            print(f"📋 Converting to: {temp_output_file}")

            # Execute conversion
            output_file = converter.convert_spans_file(spans_file, temp_output_file)

            if output_file and os.path.exists(output_file):
                # Move and enhance converted file
                enhanced_filepath = self._enhance_converted_trace_file(
                    output_file, trace_id, query, framework
                )

                # XML structure filtering (only for NexAU XML format)
                if not use_openai:
                    enhanced_filepath = self._filter_xml_errors(enhanced_filepath)
                    if not enhanced_filepath:
                        print("❌ XML validation failed - no valid records")
                        return None

                # If using NexAU format and --tool-call-format is specified, apply format conversion
                if not use_openai and self.tool_call_format:
                    enhanced_filepath = self._apply_tool_call_format_conversion(
                        enhanced_filepath, self.tool_call_format
                    )

                return enhanced_filepath
            else:
                print("❌ Conversion failed or output file not found")
                return None

        except ImportError as e:
            print(f"❌ Failed to import converter: {e}")
            import traceback

            traceback.print_exc()
            return None
        except Exception as e:
            print(f"❌ Error converting traces: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _filter_xml_errors(self, input_filepath: str) -> str:
        """
        Filter records with XML structure errors, only keep valid ones

        Args:
            input_filepath: Input file path

        Returns:
            Filtered file path, or None if no valid records
        """
        try:
            print(f"🔍 Validating XML structure...")

            # Import filter module
            sys.path.insert(0, str(self.converter_path / "trace"))
            from filter_xml_errors import filter_valid_xml

            # Execute filtering (directly overwrite original file)
            total, valid, invalid, details = filter_valid_xml(
                input_filepath, input_filepath
            )

            print(f"📊 XML Validation Results:")
            print(f"  Total records: {total}")
            print(f"  ✅ Valid: {valid}")
            print(f"  ❌ Invalid: {invalid}")

            if invalid > 0 and details:
                print(f"⚠️  Invalid records details:")
                for detail in details[:5]:  # Only show first 5
                    print(f"    - {detail}")
                if len(details) > 5:
                    print(f"    ... and {len(details) - 5} more errors")

            if valid == 0:
                print("❌ No valid XML records found")
                return None

            return input_filepath

        except Exception as e:
            print(f"⚠️  XML validation error: {e}")
            import traceback

            traceback.print_exc()
            # If filtering fails, still return original file path (degraded handling)
            return input_filepath

    def _apply_tool_call_format_conversion(
        self, input_filepath: str, target_format: str
    ) -> str:
        """
        Apply tool call format conversion (NexAU -> specific format)

        Args:
            input_filepath: Input NexAU format file path
            target_format: Target format (qwen, minimax, glm, openrouter, deepseek)

        Returns:
            Converted file path
        """
        try:
            print(f"🔧 Converting tool calls to {target_format.upper()} format...")

            # Import format conversion module
            sys.path.insert(0, str(self.converter_path / "trace"))
            from convert_trace_to_specific_tool_call_style import convert_message_format

            input_path = Path(input_filepath)
            # Generate new filename with format suffix
            output_filename = (
                input_path.stem.replace(
                    "_chatcompletion", f"_chatcompletion_{target_format}"
                )
                + input_path.suffix
            )
            output_path = input_path.parent / output_filename

            # Read and convert line by line
            converted_count = 0
            with (
                open(input_path, "r", encoding="utf-8") as infile,
                open(output_path, "w", encoding="utf-8") as outfile,
            ):

                for line in infile:
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        messages = data.get("messages", [])

                        # Convert tool calls in system and assistant messages
                        for msg in messages:
                            if msg.get("role") in ["system", "assistant"] and msg.get(
                                "content"
                            ):
                                msg["content"] = convert_message_format(
                                    msg["content"], target_format
                                )

                        # Write converted data
                        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                        converted_count += 1

                    except Exception as e:
                        print(f"⚠️ Warning: Failed to convert line: {e}")
                        # Keep original line
                        outfile.write(line)

            print(f"✅ Tool call format conversion complete: {converted_count} entries")
            print(f"📁 Output: {output_path}")

            # Delete original file
            try:
                os.remove(input_filepath)
            except:
                pass

            return str(output_path)

        except Exception as e:
            print(f"❌ Error converting tool call format: {e}")
            import traceback

            traceback.print_exc()
            # Return original file path
            return input_filepath

    def _enhance_converted_trace_file(
        self, original_filepath: str, trace_id: str, query: str, framework: str
    ) -> str:
        """Enhance converted trace file by adding query, framework, langfuse_trace_id, and stop fields (in-place modification)"""
        try:
            print(f"📝 Enhancing: {original_filepath}")

            # Read original file content
            enhanced_lines = []
            with open(original_filepath, "r", encoding="utf-8") as infile:
                for line in infile:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            # Add extra fields
                            data["query"] = query
                            data["framework"] = framework
                            data["langfuse_trace_id"] = trace_id
                            data["stop"] = (
                                ""  # Empty string indicates completely normal execution completion
                            )
                            enhanced_lines.append(
                                json.dumps(data, ensure_ascii=False) + "\n"
                            )
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipping invalid JSON line: {e}")
                            continue

            # Write back to file in-place
            with open(original_filepath, "w", encoding="utf-8") as outfile:
                outfile.writelines(enhanced_lines)

            print(f"✅ Enhanced: {original_filepath}")
            return original_filepath

        except Exception as e:
            print(f"❌ Error enhancing converted trace file: {e}")
            return original_filepath

    def _process_single_query(
        self,
        i: int,
        query_data: Dict[str, str],
        delay_seconds: int,
        total_queries: int,
        only_run_query: bool = False,
    ) -> Dict[str, Any]:
        """Process complete workflow for a single query (with log redirection to file, console silent)

        Args:
            only_run_query: If True, only run query, do not fetch and convert trace

        Returns:
            Dict containing: success(bool), result(dict or error info), query_index(int)
        """
        current_query_index = i + 1
        query = query_data.get("query", "")
        framework = query_data.get("framework", "deer-flow")

        # Generate log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_filename = f"query_{current_query_index:03d}_{framework}_{timestamp}.log"
        log_file_path = self.log_dir / log_filename

        # Open log file and pass it to sub-methods
        # Use w+ mode to allow read/write
        # Do not modify global sys.stdout/stderr to avoid multithreading conflicts
        with open(log_file_path, "w+", encoding="utf-8", buffering=1) as log_file:
            try:
                # Record start
                log_file.write(f"{'='*80}\n")
                log_file.write(
                    f"Query {current_query_index}/{total_queries} - Log Started\n"
                )
                log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                log_file.write(f"Framework: {framework}\n")
                log_file.write(
                    f"Query: {query[:200]}{'...' if len(query) > 200 else ''}\n"
                )
                log_file.write(f"{'='*80}\n\n")
                log_file.flush()

                # Execute actual processing logic, passing in log_file
                result = self._process_single_query_impl(
                    i,
                    query_data,
                    delay_seconds,
                    total_queries,
                    only_run_query,
                    log_file,
                )

                # Record end
                log_file.write(f"\n{'='*80}\n")
                log_file.write(f"Query {current_query_index} - Log Ended\n")
                log_file.write(
                    f"Status: {'✅ Success' if result.get('success') else '❌ Failed'}\n"
                )
                log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                log_file.write(f"{'='*80}\n")
                log_file.flush()

                return result

            except Exception as e:
                # Write error message to log file
                log_file.write(
                    f"\n❌ Error processing query {current_query_index}: {e}\n"
                )
                import traceback

                traceback.print_exc(file=log_file)
                log_file.flush()

                # Return failure result
                return {
                    "success": False,
                    "query_index": i,
                    "query_data": query_data,
                    "reason": f"Exception: {str(e)}",
                    "trace_id": None,
                    "stop": "exception",
                }

    def _process_single_query_impl(
        self,
        i: int,
        query_data: Dict[str, str],
        delay_seconds: int,
        total_queries: int,
        only_run_query: bool = False,
        log_file=None,
    ) -> Dict[str, Any]:
        """Actual implementation logic for processing a single query

        Args:
            log_file: Log file object, if provided all output will be written to this file
        """

        def log(msg):
            """Helper function: write to log"""
            if log_file:
                log_file.write(msg + "\n")
                log_file.flush()

        current_query_index = i + 1
        query = query_data.get("query", "")
        framework = query_data.get("framework", "deer-flow")

        # Empty query check
        if not query:
            return self._make_error_result(
                i, query_data, "Empty query", stop="empty_query"
            )

        current_trace_id = None
        current_stop_stage = ""

        try:
            log(
                f"\n🔄 [Thread-{threading.current_thread().name}] Processing Query {current_query_index}/{total_queries}"
            )
            log("-" * 40)

            # 1. Run agent query
            current_stop_stage = "agent_execution"
            agent_result = self.run_agent_query(query, framework, log_file)
            current_trace_id = agent_result.get("trace_id")

            # Check agent execution and trace_id
            if not agent_result.get("success", False):
                return self._make_error_result(
                    i,
                    query_data,
                    f"Agent execution failed: {agent_result.get('error', 'Unknown error')}",
                    current_trace_id,
                    current_stop_stage,
                )

            if not current_trace_id:
                return self._make_error_result(
                    i,
                    query_data,
                    "No trace_id found",
                    current_trace_id,
                    current_stop_stage,
                )

            # If only running query, stop here
            if only_run_query:
                log(
                    f"✅ Query {current_query_index} executed successfully (trace_id: {current_trace_id})"
                )
                log(
                    f"⏭️  Skipping trace retrieval and conversion (--only-run-query mode)"
                )
                return {
                    "success": True,
                    "query_index": i,
                    "query_data": query_data,
                    "trace_id": current_trace_id,
                    "completed_at": datetime.now().isoformat(),
                    "stop": "",
                    "only_run_query": True,
                }

            # 2. Wait for trace recording
            log(f"⏳ Waiting {delay_seconds} seconds for trace recording...")
            time.sleep(delay_seconds)

            # 3. Get traces
            current_stop_stage = "trace_retrieval"
            spans_file = self.get_traces_by_id(current_trace_id, query, framework)
            if not spans_file:
                return self._make_error_result(
                    i,
                    query_data,
                    "Trace retrieval failed",
                    current_trace_id,
                    current_stop_stage,
                )

            # 4. Convert traces
            current_stop_stage = "trace_conversion"
            converted_file = self.convert_traces(
                spans_file, framework, query, current_trace_id
            )
            if not converted_file:
                return self._make_error_result(
                    i,
                    query_data,
                    "Conversion failed",
                    current_trace_id,
                    current_stop_stage,
                )

            # Successfully completed
            log(f"✅ Query {current_query_index} completed successfully")
            log(f"📁 Final output: {converted_file}")

            return {
                "success": True,
                "query_index": i,
                "query_data": query_data,
                "trace_id": current_trace_id,
                "langfuse_trace_file": spans_file,
                "converted_trace_file": converted_file,
                "completed_at": datetime.now().isoformat(),
                "stop": "",
            }

        except Exception as e:
            log(f"❌ Error processing query {current_query_index}: {e}")
            import traceback

            traceback.print_exc(file=log_file if log_file else None)
            return self._make_error_result(
                i,
                query_data,
                f"Exception: {str(e)}",
                current_trace_id,
                current_stop_stage or "exception",
            )

    def run_end_to_end(
        self,
        max_queries: int = None,
        delay_seconds: int = 10,
        target_frameworks: List[str] = None,
        max_workers: int = 3,
        only_run_query: bool = False,
        use_openai_format: bool = False,
        tool_call_format: str = None,
    ):
        """Run end-to-end workflow with parallel processing support

        Args:
            only_run_query: If True, only run query, do not fetch and convert trace
        """
        # Save format parameters as instance variables for use by _convert_traces
        self.use_openai_format = use_openai_format
        self.tool_call_format = tool_call_format

        print("🎯 Starting NexGAP End-to-End Process")
        print("=" * 60)
        print(
            f"📋 Output format: {'OpenAI tool calls' if use_openai_format else 'NexAU XML'}"
        )
        if not use_openai_format and tool_call_format:
            print(f"🔧 Tool call format: {tool_call_format.upper()}")

        # Show available frameworks
        available_frameworks = self.get_available_frameworks()
        if available_frameworks:
            print(f"📋 Available frameworks: {', '.join(available_frameworks)}")

        # 1. Load queries
        queries = self.load_queries(target_frameworks)
        if not queries:
            print("❌ No queries found")
            return False

        # Limit query count
        if max_queries:
            queries = queries[:max_queries]
            print(f"📊 Limited to first {max_queries} queries")

        total_queries = len(queries)

        # Initialize
        completed_queries = []
        failed_queries = []

        print(f"⚡ Parallel processing enabled: max_workers={max_workers}")

        # Show execution mode
        if only_run_query:
            print("🚀 Mode: Only run queries (skip trace retrieval and conversion)")
        else:
            print("🔄 Mode: Full pipeline (query → trace retrieval → conversion)")

        # All queries need to be processed
        queries_to_process = [(i, queries[i]) for i in range(total_queries)]

        print(f"📊 To process: {len(queries_to_process)} queries")
        print(f"📊 Already completed: 0 queries")

        # 2. Process queries in parallel
        success_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_query = {
                executor.submit(
                    self._process_single_query,
                    i,
                    query_data,
                    delay_seconds,
                    total_queries,
                    only_run_query,
                ): (i, query_data)
                for i, query_data in queries_to_process
            }

            # Process completed tasks
            try:
                with tqdm(
                    total=len(queries_to_process),
                    desc="Processing queries",
                    unit="query",
                ) as pbar:
                    for future in as_completed(future_to_query):
                        result = future.result()

                        # Use lock to protect shared state updates
                        with self._lock:
                            if result["success"]:
                                completed_queries.append(result)
                                success_count += 1
                            else:
                                failed_queries.append(result)

                # Update progress bar
                pbar.update(1)

            except KeyboardInterrupt:
                print("\n⏸️ Process interrupted by user")
                # ThreadPoolExecutor will wait for submitted tasks to complete on exit
                executor.shutdown(wait=False, cancel_futures=True)

        # 3. Summary
        print("\n" + "=" * 60)
        print("📊 End-to-End Process Summary")
        print(f"✅ Successful: {success_count}/{total_queries}")
        print(f"❌ Failed: {len(failed_queries)}/{total_queries}")
        print(f"📈 Success Rate: {success_count/total_queries*100:.1f}%")

        if failed_queries:
            print(f"\n❌ Failed queries:")
            for failure in failed_queries[-3:]:  # Only show last 3 failures
                query_data = failure.get("query_data", {})
                query_text = query_data.get("query", "")[:50] + (
                    "..." if len(query_data.get("query", "")) > 50 else ""
                )
                print(
                    f"  - Query {failure['query_index']+1}: {query_text} - {failure['reason']}"
                )

        return success_count > 0


def main():
    parser = argparse.ArgumentParser(description="NexGAP End-to-End Pipeline")
    parser.add_argument(
        "--query-filepath", type=str, required=True, help="Path to query JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for all generated files (default: ./output)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Maximum number of queries to process (default: all)",
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        nargs="+",
        help="Specific frameworks to process (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=10,
        help="Seconds to wait after each query (default: 10)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers (default: 3)",
    )
    parser.add_argument(
        "--only-run-query",
        action="store_true",
        help="Only run queries without retrieving or converting traces",
    )
    parser.add_argument(
        "--use-openai-format",
        action="store_true",
        help="Convert to OpenAI tool call format (default: NexAU XML format)",
    )
    parser.add_argument(
        "--tool-call-format",
        type=str,
        choices=["qwen", "minimax", "glm", "openrouter", "deepseek"],
        help="Convert NexAU tool calls to specific format (only when using NexAU, incompatible with --use-openai-format)",
    )

    args = parser.parse_args()

    # Check parameter conflicts
    if args.use_openai_format and args.tool_call_format:
        print("❌ Error: --use-openai-format and --tool-call-format are incompatible")
        print("   --tool-call-format only applies to NexAU format conversions")
        sys.exit(1)

    try:
        runner = NexGAPRunner(
            query_file=args.query_filepath, output_dir=args.output_dir
        )

        # Check if query file exists
        if not runner.query_file.exists():
            print(f"❌ Query file not found: {runner.query_file}")
            print("   Please check the specified query file path")
            sys.exit(1)

        # Show available frameworks
        available_frameworks = runner.get_available_frameworks()
        if available_frameworks:
            print(f"📋 Available frameworks: {', '.join(available_frameworks)}")

        # Validate specified frameworks
        target_frameworks = None
        if args.frameworks:
            invalid_frameworks = [
                f for f in args.frameworks if f not in available_frameworks
            ]
            if invalid_frameworks:
                print(f"❌ Invalid frameworks: {', '.join(invalid_frameworks)}")
                print(f"📋 Available frameworks: {', '.join(available_frameworks)}")
                sys.exit(1)
            target_frameworks = args.frameworks
            print(f"🎯 Target frameworks: {', '.join(target_frameworks)}")

        success = runner.run_end_to_end(
            max_queries=args.max_queries,
            delay_seconds=args.delay,
            target_frameworks=target_frameworks,
            max_workers=args.max_workers,
            only_run_query=args.only_run_query,
            use_openai_format=args.use_openai_format,
            tool_call_format=args.tool_call_format,
        )

        if success:
            print("🎉 End-to-end process completed with some successes!")
            sys.exit(0)
        else:
            print("💥 End-to-end process failed completely!")
            sys.exit(1)

    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
