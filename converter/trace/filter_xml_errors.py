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
XML Structure Validation Filter - Only keep ChatCompletion records with correct XML structure
"""

import json
import re
from pathlib import Path
from typing import Callable, List, Tuple


class XMLValidator:
    """XML tool call format validator - supports tool_use, parallel, batch agent and other structures"""

    MODE_ALIASES = {
        "a4a": "a4a",
        "nexau": "nexau",
    }

    def __init__(self, mode: str = "a4a"):
        """Initialize validator

        Args:
            mode: Validation mode, 'a4a' or 'nexau'
        """
        normalized = (mode or "").strip().lower()
        if normalized not in self.MODE_ALIASES:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = self.MODE_ALIASES[normalized]

    def _is_agent_tool_name(self, tool_name: str | None) -> bool:
        """Check if tool_name is an agent call (starts with 'agent:')"""
        if not tool_name:
            return False
        return tool_name.strip().startswith("agent:")

    def _require_agent_message(
        self,
        *,
        container_desc: str,
        parameter_content: str,
        errors: List[str],
    ) -> None:
        """In nexau mode, agent calls must contain <message> tag"""
        if self.mode != "nexau":
            return
        if not re.search(r"<message>.*?</message>", parameter_content, re.DOTALL):
            errors.append(f"{container_desc} agent call missing <message>...</message>")

    def check_tags_balanced(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check if all XML tags in text are properly closed
        Use stack to match opening and closing tags
        Returns: (is_balanced, errors)
        """
        # First remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # Match tags: supports letters, numbers, underscores, Chinese characters, hyphens, colons
        # Also supports self-closing tags like <br/>
        tag_pattern = (
            r"<(/?)([a-zA-Z\u4e00-\u9fff][\w\u4e00-\u9fff:-]*)(?:\s+[^>]*)?\s*(/?)>"
        )
        tags = re.finditer(tag_pattern, text)

        stack: List[Tuple[str, int]] = []
        errors: List[str] = []

        # HTML self-closing tag list
        self_closing_html = {"br", "img", "hr", "input", "meta", "link"}

        for match in tags:
            is_closing = match.group(1) == "/"
            tag_name = match.group(2)
            is_self_closing = match.group(3) == "/"
            position = match.start()

            # Self-closing tags don't need to be pushed to stack
            if is_self_closing:
                continue

            # HTML self-closing tags don't need closing even without /
            if tag_name.lower() in self_closing_html and not is_closing:
                continue

            if not is_closing:
                stack.append((tag_name, position))
            else:
                if not stack:
                    errors.append(
                        f"Unmatched closing tag </{tag_name}> at position {position}"
                    )
                else:
                    top_name, top_position = stack.pop()
                    if top_name != tag_name:
                        errors.append(
                            f"Tag mismatch: <{top_name}> (position {top_position}) vs </{tag_name}> (position {position})"
                        )

        for name, position in stack:
            errors.append(f"Unclosed tag <{name}> at position {position}")

        return len(errors) == 0, errors

    def validate_tool_use_blocks(self, content: str) -> Tuple[bool, List[str]]:
        """Validate that <tool_use> blocks contain tool_name and parameter with properly closed tags"""
        errors: List[str] = []
        pattern = r"<tool_use>(.*?)</tool_use>"
        matches = list(re.finditer(pattern, content, re.DOTALL))

        if not matches:
            if "<tool_use>" in content:
                errors.append("Found unclosed <tool_use> tag")
            return len(errors) == 0, errors

        for idx, match in enumerate(matches, 1):
            block = match.group(1)
            position = match.start()

            # Check tool_name
            tool_name_match = re.search(
                r"<tool_name>(.*?)</tool_name>", block, re.DOTALL
            )
            if not tool_name_match:
                errors.append(
                    f"tool_use block #{idx} (position {position}) missing <tool_name>...</tool_name>"
                )
                tool_name_value = ""
            else:
                tool_name_value = tool_name_match.group(1).strip()

            # Check parameter
            parameter_match = re.search(
                r"<parameter>(.*?)</parameter>", block, re.DOTALL
            )
            if not parameter_match:
                errors.append(
                    f"tool_use block #{idx} (position {position}) missing <parameter>...</parameter>"
                )
                continue

            # Recursively check tag balance inside parameter
            parameter_content = parameter_match.group(1)
            balanced, balance_errors = self.check_tags_balanced(parameter_content)
            if not balanced:
                errors.append(
                    f"tool_use block #{idx} (position {position}) parameter internal tag mismatch"
                )
                errors.extend(f"  └─ {err}" for err in balance_errors)

            # nexau mode: agent calls need message tag
            if self.mode == "nexau" and self._is_agent_tool_name(tool_name_value):
                self._require_agent_message(
                    container_desc=f"tool_use block #{idx} (position {position})",
                    parameter_content=parameter_content,
                    errors=errors,
                )

        return len(errors) == 0, errors

    def validate_parallel_tool_calls(self, content: str) -> Tuple[bool, List[str]]:
        """Validate parallel_tool sub-item structure inside <use_parallel_tool_calls> block"""
        errors: List[str] = []
        block_pattern = r"<use_parallel_tool_calls>(.*?)</use_parallel_tool_calls>"
        blocks = list(re.finditer(block_pattern, content, re.DOTALL))

        if not blocks:
            if "<use_parallel_tool_calls>" in content:
                errors.append("Found unclosed <use_parallel_tool_calls> tag")
            return len(errors) == 0, errors

        for block_idx, block_match in enumerate(blocks, 1):
            block_content = block_match.group(1)
            block_position = block_match.start()

            parallel_tools = list(
                re.finditer(
                    r"<parallel_tool>(.*?)</parallel_tool>", block_content, re.DOTALL
                )
            )
            if not parallel_tools:
                errors.append(
                    f"use_parallel_tool_calls block #{block_idx} (position {block_position}) missing <parallel_tool>...</parallel_tool>"
                )
                continue

            # Check for unclosed parallel_tool tags
            start_tag_count = len(re.findall(r"<parallel_tool>", block_content))
            if start_tag_count > len(parallel_tools):
                errors.append(
                    f"use_parallel_tool_calls block #{block_idx} (position {block_position}) has unclosed <parallel_tool> tag"
                )

            for tool_idx, tool_match in enumerate(parallel_tools, 1):
                tool_content = tool_match.group(1)
                tool_position = block_position + tool_match.start()

                # Check tool_name
                tool_name_match = re.search(
                    r"<tool_name>(.*?)</tool_name>", tool_content, re.DOTALL
                )
                if not tool_name_match:
                    errors.append(
                        f"parallel_tool #{tool_idx} (position {tool_position}) missing <tool_name>...</tool_name>"
                    )
                    tool_name_value = ""
                else:
                    tool_name_value = tool_name_match.group(1).strip()

                # Check parameter
                parameter_match = re.search(
                    r"<parameter>(.*?)</parameter>", tool_content, re.DOTALL
                )
                if not parameter_match:
                    errors.append(
                        f"parallel_tool #{tool_idx} (position {tool_position}) missing <parameter>...</parameter>"
                    )
                    continue

                # Recursively check tag balance inside parameter
                parameter_content = parameter_match.group(1)
                balanced, balance_errors = self.check_tags_balanced(parameter_content)
                if not balanced:
                    errors.append(
                        f"parallel_tool #{tool_idx} (position {tool_position}) parameter internal tag mismatch"
                    )
                    errors.extend(f"  └─ {err}" for err in balance_errors)

                # nexau mode: agent calls need message tag
                if self.mode == "nexau" and self._is_agent_tool_name(tool_name_value):
                    self._require_agent_message(
                        container_desc=f"parallel_tool #{tool_idx} (position {tool_position})",
                        parameter_content=parameter_content,
                        errors=errors,
                    )

        return len(errors) == 0, errors

    def validate_parallel_sub_agents(self, content: str) -> Tuple[bool, List[str]]:
        """Validate parallel_agent and parallel_tool structure inside <use_parallel_sub_agents> block"""
        errors: List[str] = []
        block_pattern = r"<use_parallel_sub_agents>(.*?)</use_parallel_sub_agents>"
        blocks = list(re.finditer(block_pattern, content, re.DOTALL))

        if not blocks:
            if "<use_parallel_sub_agents>" in content:
                errors.append("Found unclosed <use_parallel_sub_agents> tag")
            return len(errors) == 0, errors

        for block_idx, block_match in enumerate(blocks, 1):
            block_content = block_match.group(1)
            block_position = block_match.start()

            agents = list(
                re.finditer(
                    r"<parallel_agent>(.*?)</parallel_agent>", block_content, re.DOTALL
                )
            )
            tools = list(
                re.finditer(
                    r"<parallel_tool>(.*?)</parallel_tool>", block_content, re.DOTALL
                )
            )

            if not agents and not tools:
                errors.append(
                    f"use_parallel_sub_agents block #{block_idx} (position {block_position}) missing parallel_agent/parallel_tool sub-blocks"
                )

            # Check for unclosed tags
            if len(re.findall(r"<parallel_agent>", block_content)) > len(agents):
                errors.append(
                    f"use_parallel_sub_agents block #{block_idx} (position {block_position}) has unclosed <parallel_agent> tag"
                )

            if len(re.findall(r"<parallel_tool>", block_content)) > len(tools):
                errors.append(
                    f"use_parallel_sub_agents block #{block_idx} (position {block_position}) has unclosed <parallel_tool> tag"
                )

            # Validate parallel_agent
            for agent_idx, agent_match in enumerate(agents, 1):
                agent_content = agent_match.group(1)
                agent_position = block_position + agent_match.start()

                if not re.search(
                    r"<agent_name>.*?</agent_name>", agent_content, re.DOTALL
                ):
                    errors.append(
                        f"parallel_agent #{agent_idx} (position {agent_position}) missing <agent_name>...</agent_name>"
                    )

                if not re.search(r"<message>.*?</message>", agent_content, re.DOTALL):
                    errors.append(
                        f"parallel_agent #{agent_idx} (position {agent_position}) missing <message>...</message>"
                    )

                # Validate history CDATA wrapping
                history_match = re.search(
                    r"<history>(.*?)</history>", agent_content, re.DOTALL
                )
                if history_match:
                    history_content = history_match.group(1)
                    if (
                        "<![CDATA[" not in history_content
                        or "]]>" not in history_content
                    ):
                        errors.append(
                            f"parallel_agent #{agent_idx} (position {agent_position}) history not wrapped in <![CDATA[...]]>"
                        )

            # Validate parallel_tool
            for tool_idx, tool_match in enumerate(tools, 1):
                tool_content = tool_match.group(1)
                tool_position = block_position + tool_match.start()

                if not re.search(
                    r"<tool_name>.*?</tool_name>", tool_content, re.DOTALL
                ):
                    errors.append(
                        f"parallel_tool #{tool_idx} (position {tool_position}) missing <tool_name>...</tool_name>"
                    )

                parameter_match = re.search(
                    r"<parameter>(.*?)</parameter>", tool_content, re.DOTALL
                )
                if not parameter_match:
                    errors.append(
                        f"parallel_tool #{tool_idx} (position {tool_position}) missing <parameter>...</parameter>"
                    )
                    continue

                # Recursively check tag balance inside parameter
                parameter_content = parameter_match.group(1)
                balanced, balance_errors = self.check_tags_balanced(parameter_content)
                if not balanced:
                    errors.append(
                        f"parallel_tool #{tool_idx} (position {tool_position}) parameter internal tag mismatch"
                    )
                    errors.extend(f"  └─ {err}" for err in balance_errors)

        return len(errors) == 0, errors

    def validate_sub_agent_blocks(self, content: str) -> Tuple[bool, List[str]]:
        """Validate <sub-agent> block's agent_name, message and optional history tags"""
        errors: List[str] = []
        pattern = r"<sub-agent>(.*?)</sub-agent>"
        blocks = list(re.finditer(pattern, content, re.DOTALL))

        if not blocks:
            if "<sub-agent>" in content:
                errors.append("Found unclosed <sub-agent> tag")
            return len(errors) == 0, errors

        for block_idx, block_match in enumerate(blocks, 1):
            block_content = block_match.group(1)
            block_position = block_match.start()

            if not re.search(r"<agent_name>.*?</agent_name>", block_content, re.DOTALL):
                errors.append(
                    f"sub-agent block #{block_idx} (position {block_position}) missing <agent_name>...</agent_name>"
                )

            if not re.search(r"<message>.*?</message>", block_content, re.DOTALL):
                errors.append(
                    f"sub-agent block #{block_idx} (position {block_position}) missing <message>...</message>"
                )

            # Validate history CDATA wrapping
            history_match = re.search(
                r"<history>(.*?)</history>", block_content, re.DOTALL
            )
            if history_match:
                history_content = history_match.group(1)
                if "<![CDATA[" not in history_content or "]]>" not in history_content:
                    errors.append(
                        f"sub-agent block #{block_idx} (position {block_position}) history not wrapped in <![CDATA[...]]>"
                    )

        return len(errors) == 0, errors

    def validate_batch_agent_blocks(self, content: str) -> Tuple[bool, List[str]]:
        """Validate <use_batch_agent> block structure"""
        errors: List[str] = []
        pattern = r"<use_batch_agent>(.*?)</use_batch_agent>"
        blocks = list(re.finditer(pattern, content, re.DOTALL))

        if not blocks:
            if "<use_batch_agent>" in content:
                errors.append("Found unclosed <use_batch_agent> tag")
            return len(errors) == 0, errors

        for block_idx, block_match in enumerate(blocks, 1):
            block_content = block_match.group(1)
            block_position = block_match.start()

            # Check agent_name or tool_name (depending on mode)
            if self.mode == "nexau":
                tool_name_match = re.search(
                    r"<tool_name>(.*?)</tool_name>", block_content, re.DOTALL
                )
                if not tool_name_match:
                    errors.append(
                        f"use_batch_agent block #{block_idx} (position {block_position}) missing <tool_name>...</tool_name>"
                    )
                elif not self._is_agent_tool_name(tool_name_match.group(1)):
                    errors.append(
                        f"use_batch_agent block #{block_idx} (position {block_position}) tool_name needs agent: prefix"
                    )
            else:
                if not re.search(
                    r"<agent_name>.*?</agent_name>", block_content, re.DOTALL
                ):
                    errors.append(
                        f"use_batch_agent block #{block_idx} (position {block_position}) missing <agent_name>...</agent_name>"
                    )

            # Check input_data_source
            input_match = re.search(
                r"<input_data_source>(.*?)</input_data_source>",
                block_content,
                re.DOTALL,
            )
            if not input_match:
                errors.append(
                    f"use_batch_agent block #{block_idx} (position {block_position}) missing <input_data_source>...</input_data_source>"
                )
            else:
                input_content = input_match.group(1)
                if not re.search(
                    r"<file_name>.*?</file_name>", input_content, re.DOTALL
                ):
                    errors.append(
                        f"use_batch_agent block #{block_idx} (position {block_position}) input_data_source missing <file_name>"
                    )
                if not re.search(r"<format>.*?</format>", input_content, re.DOTALL):
                    errors.append(
                        f"use_batch_agent block #{block_idx} (position {block_position}) input_data_source missing <format>"
                    )

            # Check message
            if not re.search(r"<message>.*?</message>", block_content, re.DOTALL):
                errors.append(
                    f"use_batch_agent block #{block_idx} (position {block_position}) missing <message>...</message>"
                )

        return len(errors) == 0, errors

    def get_validators(self) -> List[Callable[[str], Tuple[bool, List[str]]]]:
        """Return all validators (different validator list based on mode)"""
        if self.mode == "a4a":
            return [
                self.validate_tool_use_blocks,
                self.validate_sub_agent_blocks,
                self.validate_parallel_tool_calls,
                self.validate_parallel_sub_agents,
                self.validate_batch_agent_blocks,
            ]
        # nexau mode
        return [
            self.validate_tool_use_blocks,
            self.validate_parallel_tool_calls,
            self.validate_batch_agent_blocks,
        ]

    def validate_message(self, content: str) -> Tuple[bool, List[str]]:
        """Validate XML structure of a single message (run all validators)"""
        if not content or not isinstance(content, str):
            return True, []

        # If no XML tags, pass directly
        if "<" not in content:
            return True, []

        is_valid = True
        all_errors: List[str] = []

        # Run all validators
        for validator in self.get_validators():
            valid, errors = validator(content)
            if not valid:
                is_valid = False
                all_errors.extend(errors)

        return is_valid, all_errors


def filter_valid_xml(
    input_path: str, output_path: str = None
) -> Tuple[int, int, int, List[str]]:
    """
    Filter ChatCompletion records with correct XML structure

    Args:
        input_path: Input JSONL file path
        output_path: Output file path, if None will overwrite original file

    Returns:
        (total_lines, valid_lines, invalid_lines, invalid_details)
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    validator = XMLValidator()
    stats = {"total": 0, "valid": 0, "invalid": 0}
    invalid_details = []
    valid_records = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                stats["invalid"] += 1
                invalid_details.append(f"Line {line_num}: JSON parsing error - {e}")
                continue

            # Check XML structure of all assistant messages
            line_errors = []
            messages = data.get("messages", [])

            if not isinstance(messages, list):
                stats["invalid"] += 1
                invalid_details.append(f"Line {line_num}: messages is not a list")
                continue

            for msg_idx, message in enumerate(messages):
                if not isinstance(message, dict):
                    continue

                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    is_valid, errors = validator.validate_message(content)
                    if not is_valid:
                        # Only record first 2 errors to avoid excessive logs
                        line_errors.append(f"Msg{msg_idx}: {'; '.join(errors[:2])}")

            if line_errors:
                stats["invalid"] += 1
                invalid_details.append(f"Line {line_num}: {' | '.join(line_errors)}")
            else:
                valid_records.append(line)
                stats["valid"] += 1

    # Write valid records
    if valid_records:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(valid_records) + "\n")

    return stats["total"], stats["valid"], stats["invalid"], invalid_details


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python filter_xml_errors.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    total, valid, invalid, details = filter_valid_xml(input_file, output_file)

    print(f"Total lines: {total}")
    print(f"Valid lines: {valid}")
    print(f"Invalid lines: {invalid}")

    if details and invalid > 0:
        print("\nInvalid record details (first 10):")
        for detail in details[:10]:
            print(f"  - {detail}")
        if len(details) > 10:
            print(f"  ... and {len(details) - 10} more errors")
