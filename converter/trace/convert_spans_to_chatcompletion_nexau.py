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
Convert LangFuse spans to ChatCompletion API format
Transforms OpenAI generation spans to standard chat completion format with framework config support
"""

import argparse
import json
import os
import re

# Add schema path to import
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

LLM_GENERATION = (
    "anthropic.chat"
    if os.getenv("USE_ANTHROPIC_API", "").lower() == "true"
    else "OpenAI-generation"
)

print(f"Using LLM_GENERATION span name: {LLM_GENERATION}")


class SpansToChatCompletionConverter:
    """Convert LangFuse spans to ChatCompletion API request/response format"""

    def __init__(self, framework_config_path: Optional[str] = None):
        self.converted_count = 0
        self.spans_index = {}  # Index spans by span_id for quick lookup

    def _build_spans_index(self, spans_data: List[Dict[str, Any]]) -> None:
        """Build an index of spans by span_id for quick lookup"""
        self.spans_index = {
            span.get("span_id"): span for span in spans_data if span.get("span_id")
        }

    def _find_agent_name_for_span(self, span: Dict[str, Any]) -> Optional[str]:
        """Find the agent name for a given span by traversing the parent hierarchy"""
        current_span = span

        # Check if this span itself has an agent name
        span_name = current_span.get("span_name", "")

        # If this is a generation span, look at its parent
        if LLM_GENERATION in span_name:
            parent_id = current_span.get("parentObservationId")
            if parent_id and parent_id in self.spans_index:
                parent_span = self.spans_index[parent_id]
                parent_name = parent_span.get("span_name", "")
                return parent_name

        return None

    def _get_parent_span_name(self, span: Dict[str, Any]) -> Optional[str]:
        """
        Get the parent span name for OpenAI generation spans
        Simple method that just returns the immediate parent's span_name
        """
        # Only for OpenAI generation spans
        span_name = span.get("span_name", "")
        if LLM_GENERATION not in span_name:
            return None

        parent_id = span.get("parentObservationId")
        if parent_id and parent_id in self.spans_index:
            parent_span = self.spans_index[parent_id]
            return parent_span.get("span_name", "unknown")

        return None

    def _restore_xml_closing_tags(self, response: str) -> str:
        """Restore XML closing tags that may have been removed by stop sequences."""
        # Check for incomplete XML blocks and add missing closing tags
        restored_response = response

        # List of tag pairs to check (opening_tag, closing_tag)
        tag_pairs = [
            ("<tool_use>", "</tool_use>"),
            ("<sub-agent>", "</sub-agent>"),
            ("<parallel_tool>", "</parallel_tool>"),
            ("<parallel_agent>", "</parallel_agent>"),
            ("<use_parallel_tool_calls>", "</use_parallel_tool_calls>"),
            ("<use_parallel_sub_agents>", "</use_parallel_sub_agents>"),
            ("<use_batch_agent>", "</use_batch_agent>"),
        ]

        for open_tag, close_tag in tag_pairs:
            if (
                open_tag in restored_response
                and not restored_response.rstrip().endswith(close_tag)
            ):
                # Count open and close tags
                open_count = restored_response.count(open_tag)
                close_count = restored_response.count(close_tag)
                if open_count > close_count:
                    restored_response += close_tag

        return restored_response

    def convert_span_to_chatcompletion(
        self, span: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a single span to ChatCompletion format
        """
        # Only process OpenAI generation spans
        if span.get("span_type") != "GENERATION" or LLM_GENERATION not in span.get(
            "span_name", ""
        ):
            return None

        input_data = span.get("input", [])
        output = span.get("output")

        # Handle output - it can be either a dict or a list
        if isinstance(output, dict):
            output_data = output
        elif isinstance(output, list) and len(output) > 0:
            output_data = output[0]
        else:
            output_data = None

        if not input_data:
            return None

        # Find agent name for this span
        agent_name = self._find_agent_name_for_span(span)

        # Extract system message and process tool definitions
        messages = []

        for message in input_data:
            if message.get("role") == "assistant":
                content = message.get("content", "")
                message["content"] = self._restore_xml_closing_tags(content)
                messages.append(message)
            else:
                messages.append(message)

        # Build request
        request = {
            "model": span.get("model", ""),
            "messages": messages,
        }

        # Build response
        response_message = {
            "role": "assistant",
            "content": output_data.get("content", ""),
        }

        response = {
            "id": f"chatcmpl-{span.get('span_id', 'unknown')}",
            "object": "chat.completion",
            "created": (
                int(
                    span.get("startTime", "2025-09-24T00:00:00Z")
                    .replace("T", "")
                    .replace("Z", "")
                    .replace("-", "")
                    .replace(":", "")[:10]
                )
                if span.get("startTime")
                else 1727136000
            ),
            "model": span.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": "stop",
                }
            ],
            "usage": span.get("usage", {}),
        }

        return {
            "request": request,
            "response": response,
            "span_id": span.get("span_id"),
            "trace_id": span.get("trace_id"),
            "agent_name": agent_name,
            # "original_span": span
        }

    def _filter_last_openai_generations(
        self, spans_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter spans to keep only the last OpenAI generation for each parent span group.
        Groups spans by their parent (or by themselves if they have no parent) and keeps
        only the chronologically last OpenAI generation span in each group.
        """
        # Group OpenAI generation spans by their parent
        parent_groups = {}

        for span in spans_data:
            # Only consider OpenAI generation spans
            if span.get("span_type") == "GENERATION" and LLM_GENERATION in span.get(
                "span_name", ""
            ):
                parent_id = span.get("parentObservationId")

                # Use parent_id as the group key, or span_id if no parent
                group_key = parent_id if parent_id else span.get("span_id")

                if group_key not in parent_groups:
                    parent_groups[group_key] = []

                parent_groups[group_key].append(span)

        # For each group, keep only the last span (by startTime)
        filtered_spans = []

        for group_key, group_spans in parent_groups.items():
            if len(group_spans) == 1:
                # Only one span in group, keep it
                filtered_spans.append(group_spans[0])
            else:
                # Multiple spans, find the last one by startTime
                # Sort by startTime (handle None values)
                def get_start_time(span):
                    start_time = span.get("startTime")
                    if start_time is None:
                        return "1970-01-01T00:00:00Z"  # Default for None values
                    return start_time

                sorted_spans = sorted(group_spans, key=get_start_time)
                last_span = sorted_spans[-1]  # Get the chronologically last span
                filtered_spans.append(last_span)

                print(
                    f"📍 Group {group_key}: kept last of {len(group_spans)} OpenAI generations (span: {last_span.get('span_id')})"
                )

        return filtered_spans

    def convert_chatcompletion(self, chatcompletion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert ChatCompletion format to format that can be used directly by process.py

        Args:
            chatcompletion: ChatCompletion format data with 'request' and 'response' keys

        Returns:
            Dict with 'messages' and 'tools' keys that process.py expects
        """
        if (
            not chatcompletion
            or "request" not in chatcompletion
            or "response" not in chatcompletion
        ):
            raise ValueError(
                "Invalid ChatCompletion format: missing 'request' or 'response'"
            )

        request = chatcompletion["request"]
        response = chatcompletion["response"]

        # Extract messages from request
        messages = []
        if "messages" in request:
            messages.extend(request["messages"])

        # Extract response message from response
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice:
                response_message = choice["message"].copy()

                # Ensure the response message has the correct structure
                if "role" not in response_message:
                    response_message["role"] = "assistant"

                messages.append(response_message)

        # Preserve metadata from original chatcompletion
        result = {"messages": messages}

        # Extract tools from request if available
        if "request" in chatcompletion:
            request = chatcompletion["request"]
            if "tools" in request and request["tools"]:
                result["tools"] = request["tools"]

        # Add agent_name, trace_id, span_id if available
        if "agent_name" in chatcompletion:
            result["agent_name"] = chatcompletion["agent_name"]
        if "trace_id" in chatcompletion:
            result["trace_id"] = chatcompletion["trace_id"]
        if "span_id" in chatcompletion:
            result["span_id"] = chatcompletion["span_id"]

        return result

    def convert_spans_file(self, input_file: str, output_file: str = None) -> str:
        """
        Convert spans JSONL file to ChatCompletion format
        """
        if not output_file:
            output_file = input_file.replace(".jsonl", "_chatcompletion.jsonl")

        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print(f"🔄 Converting spans from: {input_file}")
        print(f"📤 Output to: {output_file}")

        # First pass: load all spans and build index
        spans_data = []
        with open(input_path, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    span = json.loads(line.strip())
                    spans_data.append(span)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping line {line_num}: Invalid JSON - {e}")

        # Build spans index for quick lookup
        self._build_spans_index(spans_data)
        print(f"📋 Built index with {len(self.spans_index)} spans")

        # Filter to get only the last OpenAI generation per parent span
        filtered_spans = self._filter_last_openai_generations(spans_data)
        print(
            f"🔍 Filtered to {len(filtered_spans)} last OpenAI generations from span groups"
        )

        # Second pass: convert spans
        converted_count = 0
        with open(output_file, "w", encoding="utf-8") as outfile:
            for span in filtered_spans:
                try:
                    chatcompletion = self.convert_span_to_chatcompletion(span)
                    if chatcompletion and chatcompletion.get("agent_name") != "meta":
                        converted = self.convert_chatcompletion(chatcompletion)
                        outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        converted_count += 1

                        if converted_count % 10 == 0:
                            print(f"  Processed {converted_count} generations...")

                except Exception as e:
                    print(
                        f"⚠️  Error processing span {span.get('span_id', 'unknown')}: {e}"
                    )

        print(f"✅ Conversion complete!")
        print(f"📊 Statistics:")
        print(f"  - Total spans processed: {len(spans_data)}")
        print(f"  - OpenAI generations converted: {converted_count}")
        print(f"  - Output file: {output_file}")

        return output_file

    def filter_spans_file(self, input_file: str, output_file: str = None) -> str:
        """
        Filter spans to keep only the last OpenAI generation for each span group and output to file
        """
        if not output_file:
            output_file = input_file.replace(".jsonl", "_filtered.jsonl")

        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        print(f"🔄 Filtering spans from: {input_file}")
        print(f"📤 Output to: {output_file}")

        # Load all spans
        spans_data = []
        with open(input_path, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    span = json.loads(line.strip())
                    spans_data.append(span)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping line {line_num}: Invalid JSON - {e}")

        # Build spans index for quick lookup
        self._build_spans_index(spans_data)
        print(f"📋 Built index with {len(self.spans_index)} spans")

        # Filter to get only the last OpenAI generation per parent span
        filtered_spans = self._filter_last_openai_generations(spans_data)
        print(
            f"🔍 Filtered to {len(filtered_spans)} last OpenAI generations from span groups"
        )

        # Write filtered spans to output file with agent_name annotation
        agent_name_counts = {}
        with open(output_file, "w", encoding="utf-8") as outfile:
            for span in filtered_spans:
                # Get parent span name as agent_name for OpenAI generation spans
                agent_name = self._get_parent_span_name(span)
                agent_name = agent_name if agent_name else "unknown"

                # Track agent name counts for statistics
                agent_name_counts[agent_name] = agent_name_counts.get(agent_name, 0) + 1

                # Create a copy of the span with agent_name added
                enhanced_span = span.copy()
                enhanced_span["agent_name"] = agent_name

                outfile.write(json.dumps(enhanced_span, ensure_ascii=False) + "\n")

        print(f"✅ Filter complete!")
        print(f"📊 Statistics:")
        print(f"  - Total spans loaded: {len(spans_data)}")
        print(f"  - Filtered spans output: {len(filtered_spans)}")
        print(f"  - Agent distribution: {dict(agent_name_counts)}")
        print(f"  - Output file: {output_file}")

        return output_file


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Convert LangFuse spans to ChatCompletion API format"
    )
    parser.add_argument("input", help="Path to input spans JSONL file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: input_chatcompletion.jsonl or input_filtered.jsonl)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed conversion log"
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Only filter spans to keep last OpenAI generation per group, no conversion",
    )

    args = parser.parse_args()

    try:
        converter = SpansToChatCompletionConverter()

        if args.filter_only:
            # Only filter spans, don't convert
            output_file = converter.filter_spans_file(args.input, args.output)

            if args.verbose:
                print("\n📋 Filter Details:")
                print(
                    f"  - Filtered to keep only last OpenAI generation per span group"
                )
        else:
            # Regular conversion
            output_file = converter.convert_spans_file(args.input, args.output)

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
