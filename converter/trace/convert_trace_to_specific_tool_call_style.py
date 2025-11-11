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
Tool call format converter for Nexau framework.
Converts between Nexau format and various LLM provider formats (Qwen, MiniMax, GLM, OpenRouter).
"""

import re
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ToolCallFormat(Enum):
    """Supported tool call formats."""

    NEXAU = "nexau"
    QWEN = "qwen"
    MINIMAX = "minimax"
    GLM = "glm"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"


class ToolCallInfo:
    """Parsed tool call information."""

    def __init__(
        self, tool_name: str, parameters: Dict[str, str], call_type: str = "tool"
    ):
        self.tool_name = tool_name
        self.parameters = parameters
        self.call_type = call_type  # "tool" or "sub_agent"

    def __repr__(self):
        return f"ToolCallInfo(name={self.tool_name}, params={self.parameters}, type={self.call_type})"


def parse_xml_robust(xml_content: str) -> ET.Element:
    """
    Robustly parse XML content, handling potential malformed XML.

    Args:
        xml_content: XML string to parse

    Returns:
        Parsed XML Element
    """
    try:
        return ET.fromstring(f"<root>{xml_content}</root>")
    except ET.ParseError:
        # Try to fix common issues
        xml_content = xml_content.strip()
        # Remove potential incomplete closing tags
        xml_content = re.sub(r"<\/\w+\s*$", "", xml_content)
        return ET.fromstring(f"<root>{xml_content}</root>")


def parse_nexau_tool_call(xml_content: str) -> ToolCallInfo:
    """
    Parse Nexau format tool call.

    Format:
        <tool_use>
          <tool_name>tool_name</tool_name>
          <parameter>
            <param1>value1</param1>
            <param2>value2</param2>
          </parameter>
        </tool_use>

    Args:
        xml_content: XML content of tool call

    Returns:
        ToolCallInfo object with parsed information
    """
    root = parse_xml_robust(xml_content)

    # Get tool name
    tool_name_elem = root.find(".//tool_name")
    if tool_name_elem is None:
        raise ValueError("Missing tool_name in tool_use XML")
    tool_name = (tool_name_elem.text or "").strip()

    # Get parameters
    parameters = {}
    params_elem = root.find(".//parameter")
    if params_elem is not None:
        for param in params_elem:
            param_name = param.tag
            # Handle both text and CDATA content
            param_value = (
                "".join(param.itertext()).strip() if param.text or list(param) else ""
            )
            parameters[param_name] = param_value

    return ToolCallInfo(tool_name, parameters, "tool")


def parse_nexau_sub_agent(xml_content: str) -> ToolCallInfo:
    """
    Parse Nexau format sub-agent call and convert to tool call format.

    Format:
        <sub-agent>
          <agent_name>agent_name</agent_name>
          <message>task description</message>
          <history>history messages</history>
        </sub-agent>

    Args:
        xml_content: XML content of sub-agent call

    Returns:
        ToolCallInfo object with tool_name="sub-agent" and agent parameters
    """
    root = parse_xml_robust(xml_content)

    # Extract agent_name
    agent_name_elem = root.find(".//agent_name")
    if agent_name_elem is None:
        raise ValueError("Missing agent_name in sub-agent XML")
    agent_name = (agent_name_elem.text or "").strip()

    # Extract message
    message_elem = root.find(".//message")
    message = (
        "".join(message_elem.itertext()).strip() if message_elem is not None else ""
    )

    # Extract history (optional)
    history_elem = root.find(".//history")
    history = (
        "".join(history_elem.itertext()).strip() if history_elem is not None else ""
    )

    # Convert to tool call format with tool_name="sub-agent"
    parameters = {"agent_name": agent_name, "message": message}
    if history:
        parameters["history"] = history

    return ToolCallInfo("sub-agent", parameters, "sub_agent")


def parse_nexau_batch_agent(xml_content: str) -> ToolCallInfo:
    """
    Parse Nexau format batch agent call.

    Format:
        <use_batch_agent>
          <agent_name>agent_name</agent_name>
          <input_data_source>
            <file_name>/path/to/file.jsonl</file_name>
            <format>jsonl</format>
          </input_data_source>
          <message>message template</message>
        </use_batch_agent>

    Args:
        xml_content: XML content of batch agent call

    Returns:
        ToolCallInfo object representing batch agent call
    """
    root = parse_xml_robust(xml_content)

    # Extract agent_name
    agent_name_elem = root.find(".//agent_name")
    agent_name = (
        (agent_name_elem.text or "").strip() if agent_name_elem is not None else ""
    )

    # Extract input_data_source
    input_data_elem = root.find(".//input_data_source")
    input_data_source = ""
    if input_data_elem is not None:
        file_name_elem = input_data_elem.find("file_name")
        format_elem = input_data_elem.find("format")
        file_name = (
            (file_name_elem.text or "").strip() if file_name_elem is not None else ""
        )
        format_val = (
            (format_elem.text or "jsonl").strip()
            if format_elem is not None
            else "jsonl"
        )
        input_data_source = f"{file_name}|{format_val}"

    # Extract message
    message_elem = root.find(".//message")
    message = (
        "".join(message_elem.itertext()).strip() if message_elem is not None else ""
    )

    parameters = {
        "agent_name": agent_name,
        "message": message,
        "input_data_source": input_data_source,
    }

    return ToolCallInfo("batch-agent", parameters, "batch_agent")


def convert_to_qwen_format(tool_info: ToolCallInfo) -> str:
    """
    Convert to Qwen format.

    Format:
        <tool_call>
        <function=tool_name>
        <parameter=arg_name>arg_value</parameter>
        </function>
        </tool_call>
    """
    lines = ["<tool_call>", f"<function={tool_info.tool_name}>"]
    for param_name, param_value in tool_info.parameters.items():
        lines.append(f"<parameter={param_name}>{param_value}</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def convert_to_minimax_format(tool_info: ToolCallInfo) -> str:
    """
    Convert to MiniMax format.

    Format:
        <invoke name=tool_name>
        <parameter name=arg_name>arg_value</parameter>
        </invoke>
    """
    lines = [f"<invoke name={tool_info.tool_name}>"]
    for param_name, param_value in tool_info.parameters.items():
        lines.append(f"<parameter name={param_name}>{param_value}</parameter>")
    lines.append("</invoke>")
    return "\n".join(lines)


def convert_to_glm_format(tool_info: ToolCallInfo) -> str:
    """
    Convert to GLM format.

    Format:
        <tool_call>tool_name
        <arg_key>arg_name</arg_key>
        <arg_value>arg_value</arg_value>
        </tool_call>
    """
    lines = [f"<tool_call>{tool_info.tool_name}"]
    for param_name, param_value in tool_info.parameters.items():
        lines.append(f"<arg_key>{param_name}</arg_key>")
        lines.append(f"<arg_value>{param_value}</arg_value>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def convert_to_openrouter_format(tool_info: ToolCallInfo) -> str:
    """
    Convert to OpenRouter format.

    Format:
        <tool_name>
        <parameter1_name>value1</parameter1_name>
        <parameter2_name>value2</parameter2_name>
        </tool_name>
    """
    lines = [f"<{tool_info.tool_name}>"]
    for param_name, param_value in tool_info.parameters.items():
        lines.append(f"<{param_name}>{param_value}</{param_name}>")
    lines.append(f"</{tool_info.tool_name}>")
    return "\n".join(lines)


def convert_to_deepseek_format(tool_info: ToolCallInfo) -> str:
    """
    Convert to DeepSeek format.

    Format:
        <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_name<｜tool▁sep｜>{"param": "value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>

    Note: Uses special Unicode characters:
    - ｜ (U+FF5C) - Full-width vertical bar
    - ▁ (U+2581) - Lower one eighth block (represents space)
    """
    import json

    # Convert parameters to JSON string
    args_json = json.dumps(tool_info.parameters, ensure_ascii=False)

    # Build DeepSeek format with special Unicode characters
    result = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>"
        f"{tool_info.tool_name}"
        "<｜tool▁sep｜>"
        f"{args_json}"
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )

    return result


def convert_tool_call(tool_info: ToolCallInfo, target_format: ToolCallFormat) -> str:
    """
    Convert tool call to target format.

    Args:
        tool_info: Parsed tool call information
        target_format: Target format to convert to

    Returns:
        Formatted tool call string
    """
    if target_format == ToolCallFormat.QWEN:
        return convert_to_qwen_format(tool_info)
    elif target_format == ToolCallFormat.MINIMAX:
        return convert_to_minimax_format(tool_info)
    elif target_format == ToolCallFormat.GLM:
        return convert_to_glm_format(tool_info)
    elif target_format == ToolCallFormat.OPENROUTER:
        return convert_to_openrouter_format(tool_info)
    elif target_format == ToolCallFormat.DEEPSEEK:
        return convert_to_deepseek_format(tool_info)
    else:
        raise ValueError(f"Unsupported target format: {target_format}")


def convert_single_tool_call(message: str, target_format: ToolCallFormat) -> str:
    """
    Convert single tool calls in message.

    Args:
        message: Message containing Nexau format tool calls
        target_format: Target format

    Returns:
        Message with converted tool calls
    """
    # Pattern for single tool calls
    pattern = r"<tool_use>(.*?)</tool_use>"

    def replace_func(match):
        xml_content = match.group(1)
        try:
            tool_info = parse_nexau_tool_call(xml_content)
            return convert_tool_call(tool_info, target_format)
        except Exception as e:
            # If parsing fails, keep original
            return match.group(0)

    return re.sub(pattern, replace_func, message, flags=re.DOTALL)


def convert_sub_agent_calls(message: str, target_format: ToolCallFormat) -> str:
    """
    Convert sub-agent calls to tool call format.

    Args:
        message: Message containing Nexau format sub-agent calls
        target_format: Target format

    Returns:
        Message with converted sub-agent calls
    """
    # Pattern for sub-agent calls
    pattern = r"<sub-agent>(.*?)</sub-agent>"

    def replace_func(match):
        xml_content = match.group(1)
        try:
            tool_info = parse_nexau_sub_agent(xml_content)
            return convert_tool_call(tool_info, target_format)
        except Exception as e:
            # If parsing fails, keep original
            return match.group(0)

    return re.sub(pattern, replace_func, message, flags=re.DOTALL)


def convert_parallel_tool_calls(message: str, target_format: ToolCallFormat) -> str:
    """
    Convert parallel tool calls.

    Keeps <use_parallel_tool_calls> wrapper, converts each <parallel_tool> internally.

    Args:
        message: Message containing parallel tool calls
        target_format: Target format

    Returns:
        Message with converted parallel tool calls
    """
    # Pattern for parallel tool calls block
    pattern = r"<use_parallel_tool_calls>(.*?)</use_parallel_tool_calls>"

    def replace_block(match):
        block_content = match.group(1)
        # Pattern for individual parallel tools
        tool_pattern = r"<parallel_tool>(.*?)</parallel_tool>"

        def replace_tool(tool_match):
            xml_content = tool_match.group(1)
            try:
                tool_info = parse_nexau_tool_call(xml_content)
                return convert_tool_call(tool_info, target_format)
            except Exception as e:
                return tool_match.group(0)

        converted_content = re.sub(
            tool_pattern, replace_tool, block_content, flags=re.DOTALL
        )
        return f"<use_parallel_tool_calls>{converted_content}</use_parallel_tool_calls>"

    return re.sub(pattern, replace_block, message, flags=re.DOTALL)


def convert_parallel_sub_agents(message: str, target_format: ToolCallFormat) -> str:
    """
    Convert parallel sub-agents and tools.

    Keeps <use_parallel_sub_agents> wrapper, converts each <parallel_agent> and <parallel_tool>.

    Args:
        message: Message containing parallel sub-agents
        target_format: Target format

    Returns:
        Message with converted parallel sub-agents
    """
    # Pattern for parallel sub-agents block
    pattern = r"<use_parallel_sub_agents>(.*?)</use_parallel_sub_agents>"

    def replace_block(match):
        block_content = match.group(1)

        # Convert parallel agents
        agent_pattern = r"<parallel_agent>(.*?)</parallel_agent>"

        def replace_agent(agent_match):
            xml_content = agent_match.group(1)
            try:
                tool_info = parse_nexau_sub_agent(xml_content)
                return convert_tool_call(tool_info, target_format)
            except Exception as e:
                return agent_match.group(0)

        converted_content = re.sub(
            agent_pattern, replace_agent, block_content, flags=re.DOTALL
        )

        # Convert parallel tools
        tool_pattern = r"<parallel_tool>(.*?)</parallel_tool>"

        def replace_tool(tool_match):
            xml_content = tool_match.group(1)
            try:
                tool_info = parse_nexau_tool_call(xml_content)
                return convert_tool_call(tool_info, target_format)
            except Exception as e:
                return tool_match.group(0)

        converted_content = re.sub(
            tool_pattern, replace_tool, converted_content, flags=re.DOTALL
        )

        return f"<use_parallel_sub_agents>{converted_content}</use_parallel_sub_agents>"

    return re.sub(pattern, replace_block, message, flags=re.DOTALL)


def convert_batch_agent(message: str, target_format: ToolCallFormat) -> str:
    """
    Convert batch agent calls to tool call format.

    Args:
        message: Message containing batch agent calls
        target_format: Target format

    Returns:
        Message with converted batch agent calls
    """
    pattern = r"<use_batch_agent>(.*?)</use_batch_agent>"

    def replace_func(match):
        xml_content = match.group(1)
        try:
            tool_info = parse_nexau_batch_agent(xml_content)
            return convert_tool_call(tool_info, target_format)
        except Exception as e:
            return match.group(0)

    return re.sub(pattern, replace_func, message, flags=re.DOTALL)


def convert_message_format(message: str, target_format: str) -> str:
    """
    Main conversion function to convert entire message from Nexau format to target format.

    This function handles all types of tool calls:
    - Single tool calls (<tool_use>)
    - Sub-agent calls (<sub-agent>)
    - Parallel tool calls (<use_parallel_tool_calls>)
    - Parallel sub-agents (<use_parallel_sub_agents>)
    - Batch agent calls (<use_batch_agent>)

    Args:
        message: Original message in Nexau format
        target_format: Target format name ("qwen", "minimax", "glm", "openrouter")

    Returns:
        Converted message with all tool calls in target format

    Example:
        >>> nexau_msg = '''<tool_use>
        ...   <tool_name>search</tool_name>
        ...   <parameter>
        ...     <query>python tutorial</query>
        ...     <max_results>5</max_results>
        ...   </parameter>
        ... </tool_use>'''
        >>> qwen_msg = convert_message_format(nexau_msg, "qwen")
    """
    # Convert target format string to enum
    format_map = {
        "qwen": ToolCallFormat.QWEN,
        "minimax": ToolCallFormat.MINIMAX,
        "glm": ToolCallFormat.GLM,
        "openrouter": ToolCallFormat.OPENROUTER,
        "deepseek": ToolCallFormat.DEEPSEEK,
    }

    if target_format.lower() not in format_map:
        raise ValueError(
            f"Unsupported format: {target_format}. Supported: {list(format_map.keys())}"
        )

    format_enum = format_map[target_format.lower()]

    # Apply conversions in order
    # 1. Batch agents (least common, check first)
    message = convert_batch_agent(message, format_enum)

    # 2. Parallel sub-agents (includes both agents and tools)
    message = convert_parallel_sub_agents(message, format_enum)

    # 3. Parallel tool calls
    message = convert_parallel_tool_calls(message, format_enum)

    # 4. Single sub-agent calls
    message = convert_sub_agent_calls(message, format_enum)

    # 5. Single tool calls
    message = convert_single_tool_call(message, format_enum)

    return message


# Example usage and testing
if __name__ == "__main__":
    # Test single tool call
    nexau_single = """Here's my response:
<tool_use>
  <tool_name>search_web</tool_name>
  <parameter>
    <query>python tutorial</query>
    <max_results>5</max_results>
  </parameter>
</tool_use>"""

    print("=== Single Tool Call ===")
    print("Qwen format:")
    print(convert_message_format(nexau_single, "qwen"))
    print("\nMiniMax format:")
    print(convert_message_format(nexau_single, "minimax"))
    print("\nGLM format:")
    print(convert_message_format(nexau_single, "glm"))
    print("\nOpenRouter format:")
    print(convert_message_format(nexau_single, "openrouter"))
    print("\nDeepSeek format:")
    print(convert_message_format(nexau_single, "deepseek"))

    # Test sub-agent call
    nexau_subagent = """<sub-agent>
  <agent_name>engineer</agent_name>
  <message>Write a Python function</message>
  <history>[{"role": "user", "content": "Hello"}]</history>
</sub-agent>"""

    print("\n\n=== Sub-Agent Call ===")
    print("Qwen format:")
    print(convert_message_format(nexau_subagent, "qwen"))

    # Test parallel tools
    nexau_parallel = """<use_parallel_tool_calls>
<parallel_tool>
  <tool_name>tool1</tool_name>
  <parameter>
    <param1>value1</param1>
  </parameter>
</parallel_tool>
<parallel_tool>
  <tool_name>tool2</tool_name>
  <parameter>
    <param2>value2</param2>
  </parameter>
</parallel_tool>
</use_parallel_tool_calls>"""

    print("\n\n=== Parallel Tool Calls ===")
    print("Qwen format:")
    print(convert_message_format(nexau_parallel, "qwen"))
