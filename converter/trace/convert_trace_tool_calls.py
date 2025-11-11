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
Convert trace data tool calls to specific LLM formats.
Integrates with convert_spans_to_chatcompletion.py format and applies tool call style conversions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import the conversion functions
from convert_trace_to_specific_tool_call_style import convert_message_format


class TraceToolCallConverter:
    """Convert trace data files to specific tool call formats."""

    def __init__(self, target_format: str, verbose: bool = False):
        """
        Initialize converter.

        Args:
            target_format: Target format (qwen, minimax, glm, openrouter, deepseek)
            verbose: Enable verbose logging
        """
        self.target_format = target_format.lower()
        self.verbose = verbose

        # Validate format
        valid_formats = ["qwen", "minimax", "glm", "openrouter", "deepseek"]
        if self.target_format not in valid_formats:
            raise ValueError(
                f"Invalid format: {target_format}. Must be one of: {valid_formats}"
            )

        self.converted_count = 0
        self.error_count = 0
        self.total_messages = 0

    def convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Convert messages in a conversation.

        Only converts system and assistant messages (model behavior).
        User messages are kept unchanged as they are user input.

        Args:
            messages: List of message dicts with role and content

        Returns:
            List of converted messages
        """
        converted_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Only convert system and assistant messages
            # User messages are kept as-is (they are user input, not model behavior)
            if content and role in ["system", "assistant"]:
                try:
                    converted_content = convert_message_format(
                        content, self.target_format
                    )
                    self.total_messages += 1
                    converted_messages.append({**msg, "content": converted_content})
                except Exception as e:
                    self.error_count += 1
                    if self.verbose:
                        print(
                            f"   Warning: Failed to convert message: {e}",
                            file=sys.stderr,
                        )
                    # Keep original on error
                    converted_messages.append(msg)
            else:
                # Keep user messages and messages without content as-is
                converted_messages.append(msg)

        return converted_messages

    def convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """
        Convert tools definitions.

        Args:
            tools: List of tool definition dicts

        Returns:
            List of converted tool definitions
        """
        converted_tools = []
        for tool in tools:
            converted_tool = {}
            for key, value in tool.items():
                if isinstance(value, str):
                    # Convert string values
                    converted_value = convert_message_format(value, self.target_format)
                    converted_tool[key] = converted_value
                elif isinstance(value, dict):
                    # Recursively convert dict values
                    converted_tool[key] = self._convert_dict_values(value)
                else:
                    # Keep other values as-is
                    converted_tool[key] = value
            converted_tools.append(converted_tool)
        return converted_tools

    def _convert_dict_values(self, d: Dict) -> Dict:
        """Recursively convert string values in a dict."""
        result = {}
        for key, value in d.items():
            if isinstance(value, str):
                result[key] = convert_message_format(value, self.target_format)
            elif isinstance(value, dict):
                result[key] = self._convert_dict_values(value)
            elif isinstance(value, list):
                result[key] = [
                    (
                        convert_message_format(item, self.target_format)
                        if isinstance(item, str)
                        else (
                            self._convert_dict_values(item)
                            if isinstance(item, dict)
                            else item
                        )
                    )
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def convert_trace_entry(self, entry: Dict) -> Dict:
        """
        Convert a single trace entry.

        Args:
            entry: Trace entry dict with messages and tools

        Returns:
            Converted entry
        """
        messages = entry.get("messages", [])
        converted_messages = self.convert_messages(messages)

        # Also convert tools field if it exists
        tools = entry.get("tools", [])
        converted_tools = self.convert_tools(tools) if tools else tools

        return {**entry, "messages": converted_messages, "tools": converted_tools}

    def convert_file(self, input_path: str, output_path: str):
        """
        Convert an entire trace file.

        Args:
            input_path: Input JSONL file path
            output_path: Output JSONL file path
        """
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Process file line by line
        line_count = 0

        with (
            open(input_file, "r", encoding="utf-8") as infile,
            open(output_file, "w", encoding="utf-8") as outfile,
        ):

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse JSON
                    entry = json.loads(line)

                    # Convert
                    converted_entry = self.convert_trace_entry(entry)

                    # Write output
                    outfile.write(
                        json.dumps(converted_entry, ensure_ascii=False) + "\n"
                    )

                    line_count += 1
                    self.converted_count += 1

                    # Progress indicator
                    if line_count % 10 == 0:
                        print(f"  Processed {line_count} entries...", file=sys.stderr)

                except json.JSONDecodeError as e:
                    self.error_count += 1
                    if self.verbose:
                        print(
                            f"   Warning: Invalid JSON at line {line_num}: {e}",
                            file=sys.stderr,
                        )
                    continue

                except Exception as e:
                    self.error_count += 1
                    if self.verbose:
                        print(
                            f"   Warning: Error processing line {line_num}: {e}",
                            file=sys.stderr,
                        )
                    continue

    def print_stats(self, output_path: str):
        """Print conversion statistics."""
        print(f"\n✅ Conversion complete!")
        print(f"📊 Statistics:")
        print(f"  - Total lines processed: {self.converted_count}")
        print(f"  - Successfully converted: {self.converted_count}")
        print(f"  - Messages processed: {self.total_messages}")
        print(f"  - Errors encountered: {self.error_count}")
        print(f"  - Output file: {output_path}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert trace data tool calls to specific LLM formats"
    )

    parser.add_argument("input", help="Path to input trace JSONL file")

    parser.add_argument(
        "-f",
        "--format",
        required=True,
        choices=["qwen", "minimax", "glm", "openrouter", "deepseek"],
        help="Target format for tool calls",
    )

    parser.add_argument(
        "-o", "--output", help="Output file path (default: input_<format>.jsonl)"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_{args.format}.jsonl"

    # Print header
    print(f"🔄 Converting trace file: {args.input}")
    print(f"   Target format: {args.format.upper()}")
    print(f"📤 Output to: {output_path}")
    print()

    # Convert
    try:
        converter = TraceToolCallConverter(args.format, verbose=args.verbose)
        converter.convert_file(args.input, output_path)
        converter.print_stats(output_path)

        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
