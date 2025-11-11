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
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.append(str(Path(__file__).parent.parent / "schema"))

LLM_GENERATION = (
    "anthropic.chat"
    if os.getenv("USE_ANTHROPIC_API", "").lower() == "true"
    else "OpenAI-generation"
)

print(f"Using LLM_GENERATION span name: {LLM_GENERATION}")

try:
    from framework_config_schema import FrameworkAgent, FrameworkConfig, FrameworkTool
except ImportError as e:
    print(f"Warning: Could not import framework schema: {e}")
    FrameworkConfig = None
    FrameworkTool = None
    FrameworkAgent = None


class SpansToChatCompletionConverter:
    """Convert LangFuse spans to ChatCompletion API request/response format"""

    def __init__(self, framework_config_path: Optional[str] = None):
        self.framework_config = None
        self.framework_config_path = framework_config_path
        self.framework_tools = {}
        self.converted_count = 0
        self.spans_index = {}  # Index spans by span_id for quick lookup
        self.agent_tools_cache = {}  # Cache agent -> tools mapping
        self.all_tool_definitions = {}  # All tool definitions by tool name
        self.mcp_server_tools = {}  # Map MCP server name -> list of tool names

        # Load framework config if provided
        if framework_config_path:
            self.load_framework_config(framework_config_path)

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call identifier compatible with OpenAI format."""
        return f"call_{uuid.uuid4().hex}"

    def load_framework_config(self, config_path: str) -> None:
        """Load and parse framework config"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if FrameworkConfig:
                self.framework_config = FrameworkConfig(**config_data)
                # Load actual tool info from YAML files and index by actual names
                if self.framework_config.tools:
                    base_path = Path(config_path).parent
                    for framework_tool in self.framework_config.tools:
                        tool_schema = self.load_tool_schema_from_file(
                            framework_tool, base_path
                        )
                        if tool_schema:
                            # Use actual name from YAML file
                            actual_tool_name = tool_schema.get(
                                "name", framework_tool.tool_name
                            )
                            # Store the enhanced framework tool with actual info
                            enhanced_tool = type(
                                "EnhancedTool",
                                (),
                                {
                                    "tool_name": actual_tool_name,
                                    "description": tool_schema.get(
                                        "description", framework_tool.description
                                    ),
                                    "config_path": framework_tool.config_path,
                                    "binding": framework_tool.binding,
                                    "framework_tool_name": framework_tool.tool_name,  # Keep original for reference
                                },
                            )()
                            self.framework_tools[actual_tool_name] = enhanced_tool

            print(
                f"✅ Loaded framework config with {len(self.framework_tools)} tools ({list(self.framework_tools.keys())})"
            )

            # Pre-build tool definitions and agent tools cache
            self._build_tool_definitions_cache()
        except Exception as e:
            print(f"⚠️  Error loading framework config: {e}")

    def _build_tool_definitions_cache(self) -> None:
        """Pre-build all tool definitions and agent tools mapping"""
        if not self.framework_config:
            return

        base_path = (
            Path(self.framework_config_path).parent
            if self.framework_config_path
            else None
        )

        # Build all tool definitions (static tools)
        if self.framework_config.tools:
            for framework_tool in self.framework_config.tools:
                tool_def = self.create_tool_definition_from_framework_tool(
                    framework_tool, base_path
                )
                if tool_def:
                    # Use the tool name from the tool definition
                    tool_name = tool_def["function"]["name"]
                    self.all_tool_definitions[tool_name] = tool_def

        # Load MCP tools if MCP servers are configured
        if (
            hasattr(self.framework_config, "mcp_servers")
            and self.framework_config.mcp_servers
        ):
            try:
                print(
                    f"  🔌 Loading {len(self.framework_config.mcp_servers)} MCP Servers..."
                )
                # Load tools from each MCP server separately to track server->tools mapping
                for mcp_server in self.framework_config.mcp_servers:
                    server_name = (
                        mcp_server.name
                        if hasattr(mcp_server, "name")
                        else str(mcp_server)
                    )
                    server_tools = self._load_mcp_tools([mcp_server])
                    tool_names = []
                    for tool_def in server_tools:
                        tool_name = tool_def["function"]["name"]
                        self.all_tool_definitions[tool_name] = tool_def
                        tool_names.append(tool_name)
                    self.mcp_server_tools[server_name] = tool_names
                    if tool_names:
                        print(f"    ✅ {server_name}: {len(tool_names)} tools")
                total_mcp_tools = sum(
                    len(tools) for tools in self.mcp_server_tools.values()
                )
                print(f"  ✅ Total loaded {total_mcp_tools} MCP tools")
            except Exception as e:
                print(f"  ⚠️  Failed to load MCP tools: {str(e)[:100]}")

        # Build workflow graph to determine which agents can call other agents
        agent_successors = {}  # agent_id -> [successor_agent_ids]
        if (
            hasattr(self.framework_config, "workflow")
            and self.framework_config.workflow
        ):
            workflow = self.framework_config.workflow

            # Get all agent nodes
            agent_nodes = set()
            if hasattr(workflow, "nodes") and workflow.nodes:
                for node in workflow.nodes:
                    if hasattr(node, "type") and node.type == "agent":
                        agent_nodes.add(node.id)

            # Build successor mapping for agents
            agent_successors = {agent_id: [] for agent_id in agent_nodes}

            if hasattr(workflow, "edges") and workflow.edges:
                for edge in workflow.edges:
                    from_id = edge.from_
                    to_id = edge.to

                    # Only consider agent->agent relationships for sub_agents
                    if from_id in agent_nodes and to_id in agent_nodes:
                        agent_successors[from_id].append(to_id)

        # Build agent -> tools mapping
        if self.framework_config.agents:
            for agent in self.framework_config.agents:
                agent_tools = []

                # Add configured static tools for this agent
                if agent.tools:
                    for tool_name in agent.tools:
                        # tool_name here is from framework config, need to map to actual YAML tool name
                        # Find the framework tool with this name
                        framework_tool = next(
                            (
                                ft
                                for ft in self.framework_config.tools
                                if ft.tool_name == tool_name
                            ),
                            None,
                        )
                        if framework_tool:
                            # Load the actual tool name from YAML
                            tool_schema = self.load_tool_schema_from_file(
                                framework_tool, base_path
                            )
                            if tool_schema:
                                actual_tool_name = tool_schema.get(
                                    "name", framework_tool.tool_name
                                )
                                if actual_tool_name in self.all_tool_definitions:
                                    agent_tools.append(
                                        self.all_tool_definitions[actual_tool_name]
                                    )

                # Add MCP tools for this agent
                if hasattr(agent, "mcp_servers") and agent.mcp_servers:
                    for server_name in agent.mcp_servers:
                        # Get all tools from this MCP server
                        if server_name in self.mcp_server_tools:
                            for tool_name in self.mcp_server_tools[server_name]:
                                if tool_name in self.all_tool_definitions:
                                    agent_tools.append(
                                        self.all_tool_definitions[tool_name]
                                    )

                # Only add sub-agents based on workflow edges
                agent_name = agent.agent_name
                if agent_name in agent_successors and agent_successors[agent_name]:
                    # This agent has successors in the workflow, so it can call other agents
                    for successor_agent_name in agent_successors[agent_name]:
                        # Find the successor agent to create sub-agent tool
                        for other_agent in self.framework_config.agents:
                            if other_agent.agent_name == successor_agent_name:
                                sub_agent_tool = self.create_sub_agent_tool_definition_from_framework_agent(
                                    other_agent
                                )
                                if sub_agent_tool:
                                    agent_tools.append(sub_agent_tool)
                                break

                self.agent_tools_cache[agent.agent_name] = agent_tools

    def _load_mcp_tools(self, mcp_servers: List[Any]) -> List[Dict[str, Any]]:
        """Load tools from MCP servers using nexau's MCP client (with caching)"""
        try:
            import hashlib

            # Import nexau's MCP client
            import sys

            nexau_path = Path(__file__).parent.parent.parent / "NexA4A" / "nexau"
            if str(nexau_path) not in sys.path:
                sys.path.insert(0, str(nexau_path))

            # Load cache (with file locking for concurrent access)
            import fcntl

            from nexau.archs.tool.builtin.mcp_client import sync_initialize_mcp_tools

            cache_file = (
                Path(__file__).parent.parent / ".cache" / "mcp_tools_cache.json"
            )
            cache_file.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure cache directory exists

            # Read cache with shared lock
            cache = {}
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        fcntl.flock(
                            f.fileno(), fcntl.LOCK_SH
                        )  # Shared lock for reading
                        cache = json.load(f)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
                except:
                    cache = {}

            # Check cache for each server individually (matching preload logic)
            all_tools_from_cache = []
            servers_to_load = []

            for srv in mcp_servers:
                # Generate cache key for this single server (must match preload logic!)
                # Only use URL as the unique identifier
                url = srv.url if hasattr(srv, "url") else None
                if not url:
                    continue
                cache_key = hashlib.md5(url.encode()).hexdigest()

                if cache_key in cache:
                    # Cache hit - use cached tools
                    cached_tools = cache[cache_key]
                    all_tools_from_cache.extend(cached_tools)
                    print(
                        f"  ⚡ Using cache: {srv.name if hasattr(srv, 'name') else 'unknown'} ({len(cached_tools)} tools)"
                    )
                else:
                    # Cache miss - need to load this server
                    servers_to_load.append(srv)

            # If all servers are cached, return immediately
            if not servers_to_load:
                print(
                    f"  ✅ All MCP tools from cache (total {len(all_tools_from_cache)})"
                )
                return all_tools_from_cache

            # Some servers need to be loaded
            print(f"  ⚠️  Need to load {len(servers_to_load)} uncached MCP servers")
            mcp_servers = servers_to_load  # Only load uncached servers

            # Convert MCP server configs to the format expected by nexau
            server_configs = []
            for server in mcp_servers:
                config = {
                    "name": server.name if hasattr(server, "name") else str(server),
                    "type": server.type if hasattr(server, "type") else "stdio",
                }

                # Add type-specific configs
                if hasattr(server, "command"):
                    config["command"] = server.command
                if hasattr(server, "args"):
                    config["args"] = server.args
                if hasattr(server, "env"):
                    config["env"] = server.env
                if hasattr(server, "url"):
                    config["url"] = server.url
                if hasattr(server, "headers"):
                    config["headers"] = server.headers
                if hasattr(server, "timeout"):
                    config["timeout"] = server.timeout

                server_configs.append(config)

            # Load MCP tools for uncached servers
            mcp_tool_objects = sync_initialize_mcp_tools(server_configs)

            # Convert to OpenAI tools schema format
            newly_loaded_tools = []
            for tool in mcp_tool_objects:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": (
                            tool.description if hasattr(tool, "description") else ""
                        ),
                        "parameters": (
                            tool.input_schema
                            if hasattr(tool, "input_schema")
                            else {"type": "object", "properties": {}, "required": []}
                        ),
                    },
                }
                newly_loaded_tools.append(tool_def)

            # Save newly loaded tools to cache (incremental update with file locking)
            if servers_to_load and newly_loaded_tools:
                try:
                    # Note: sync_initialize_mcp_tools returns all server tools mixed together
                    # We need to save each server separately. Using simplified strategy:
                    # If only one server, save all tools; if multiple, distribute evenly

                    tools_per_server = (
                        len(newly_loaded_tools) // len(servers_to_load)
                        if len(servers_to_load) > 0
                        else 0
                    )
                    tool_index = 0

                    # Use exclusive lock for incremental update
                    with open(
                        cache_file,
                        "r+" if cache_file.exists() else "w+",
                        encoding="utf-8",
                    ) as f:
                        fcntl.flock(
                            f.fileno(), fcntl.LOCK_EX
                        )  # Exclusive lock for writing

                        # Re-read latest cache (avoid overwriting other process updates)
                        f.seek(0)
                        try:
                            current_cache = json.load(f)
                        except:
                            current_cache = {}

                        # Save tools for each newly loaded server
                        for i, srv in enumerate(servers_to_load):
                            url = srv.url if hasattr(srv, "url") else None
                            if not url:
                                continue
                            cache_key = hashlib.md5(url.encode()).hexdigest()

                            # Assign tools to this server
                            if i == len(servers_to_load) - 1:
                                # Last server gets remaining tools
                                server_tools = newly_loaded_tools[tool_index:]
                            else:
                                # Other servers get even distribution
                                server_tools = newly_loaded_tools[
                                    tool_index : tool_index + tools_per_server
                                ]
                                tool_index += tools_per_server

                            # Incremental update: only add new, don't overwrite existing
                            if cache_key not in current_cache:
                                current_cache[cache_key] = server_tools
                                print(
                                    f"  💾 Cached {srv.name if hasattr(srv, 'name') else 'unknown'}: {len(server_tools)} tools"
                                )

                        # Write back to file
                        f.seek(0)
                        f.truncate()
                        json.dump(current_cache, f, ensure_ascii=False, indent=2)

                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock

                except Exception as e:
                    print(f"  ⚠️  Failed to save cache: {str(e)[:100]}")

            # Return all tools (cached + newly loaded)
            all_tools = all_tools_from_cache + newly_loaded_tools
            return all_tools

        except ImportError as e:
            print(f"  ⚠️  Failed to import nexau MCP client: {e}")
            return []
        except Exception as e:
            print(f"  ⚠️  Error loading MCP tools: {str(e)[:200]}")
            return []

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

                # Check if parent is an agent span (case-insensitive if cache exists)
                if self.agent_tools_cache:
                    # Try exact match first
                    if parent_name in self.agent_tools_cache:
                        return parent_name
                    # Try case-insensitive match
                    for cached_name in self.agent_tools_cache.keys():
                        if cached_name.lower() == parent_name.lower():
                            return cached_name

                    # Fallback: if not in cache but parent name looks like an agent name, use it
                    # This handles sub-agents called by MetaAgent that aren't in the framework config
                    if parent_name and parent_name != LLM_GENERATION:
                        return parent_name
                else:
                    # No framework config, just return the parent name
                    if parent_name and parent_name != LLM_GENERATION:
                        return parent_name

                # If parent is something like "Sub-agent: poem_agent", extract the agent name
                if parent_name.startswith("Sub-agent: "):
                    agent_name = parent_name.replace("Sub-agent: ", "")
                    if self.agent_tools_cache:
                        if agent_name in self.agent_tools_cache:
                            return agent_name
                        # Try case-insensitive
                        for cached_name in self.agent_tools_cache.keys():
                            if cached_name.lower() == agent_name.lower():
                                return cached_name
                    else:
                        return agent_name

                # Recursively check parent's parent if needed
                return self._find_agent_name_for_span(parent_span)

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

    def load_tool_schema_from_file(
        self, tool: FrameworkTool, base_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Load tool schema from YAML file"""
        try:
            tool_path = base_path / tool.config_path
            if not tool_path.exists():
                # Try alternative paths
                alt_paths = [
                    base_path / "tools" / f"{tool.tool_name}.yaml",
                    base_path / "tools" / f"{tool.tool_name}.yml",
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        tool_path = alt_path
                        break

            if tool_path.exists():
                with open(tool_path, "r", encoding="utf-8") as f:
                    tool_schema = yaml.safe_load(f)
                return tool_schema
        except Exception as e:
            print(f"⚠️  Error loading tool schema for {tool.tool_name}: {e}")
        return None

    def extract_tool_definitions_from_schema(
        self,
        system_content: str,
        base_path: Optional[Path] = None,
        agent_name: Optional[str] = None,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Extract tool definitions from framework schema and clean system prompt
        Returns (cleaned_sysprompt, tool_definitions)
        """
        tools = []
        cleaned_content = system_content

        # Get tools for this specific agent from cache
        if agent_name and agent_name in self.agent_tools_cache:
            tools = self.agent_tools_cache[agent_name].copy()
        elif not agent_name and self.framework_config:
            # Fallback: if no agent name provided, include all tools and agents (backward compatibility)
            # This shouldn't happen with the new approach, but keep for safety
            if self.framework_config.tools:
                for framework_tool in self.framework_config.tools:
                    tool_definition = self.create_tool_definition_from_framework_tool(
                        framework_tool, base_path
                    )
                    if tool_definition:
                        tools.append(tool_definition)

            if self.framework_config.agents:
                for framework_agent in self.framework_config.agents:
                    sub_agent_tool = (
                        self.create_sub_agent_tool_definition_from_framework_agent(
                            framework_agent
                        )
                    )
                    if sub_agent_tool:
                        tools.append(sub_agent_tool)

        # Clean system prompt by removing tool and sub-agent documentation sections
        # Remove tool definitions section
        tool_section_patterns = [
            r"<TOOL_DEFINITIONS_START>.*?<TOOL_DEFINITIONS_END>",
            r"## Available Tools\s*\n.*?(?=## Available Sub-Agents|## Available Sub-agents|$)",
        ]

        for pattern in tool_section_patterns:
            tool_match = re.search(pattern, cleaned_content, re.DOTALL)
            if tool_match:
                cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.DOTALL)
                break

        # Remove sub-agents documentation section
        sub_agent_section_patterns = [
            r"<SUB_AGENTS_DEFINITIONS_START>.*?<SUB_AGENTS_DEFINITIONS_END>",
            r"## Available Sub-Agents.*?(?=\nWhen you use tools|\nFor parallel execution|\nFor batch processing|$)",
        ]

        for pattern in sub_agent_section_patterns:
            sub_agent_match = re.search(pattern, cleaned_content, re.DOTALL)
            if sub_agent_match:
                cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.DOTALL)
                break

        # Remove tool usage instructions section
        usage_instructions_patterns = [
            r"<TOOL_USAGE_INSTRUCTIONS_START>.*?<TOOL_USAGE_INSTRUCTIONS_END>",
            r"\nWhen you use tools or sub-agents.*?(?=\n\n[A-Z]|\n\nIMPORTANT:|$)",
        ]

        for pattern in usage_instructions_patterns:
            usage_match = re.search(pattern, cleaned_content, re.DOTALL)
            if usage_match:
                cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.DOTALL)
                break

        return cleaned_content, tools

    def create_tool_definition_from_framework_tool(
        self, framework_tool, base_path: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """Create JSON Schema tool definition from FrameworkTool"""
        try:
            # Load tool schema from YAML file
            tool_schema = self.load_tool_schema_from_file(framework_tool, base_path)

            if tool_schema and "input_schema" in tool_schema:
                return {
                    "type": "function",
                    "function": {
                        "name": tool_schema.get("name", framework_tool.tool_name),
                        "description": tool_schema.get("description", ""),
                        "parameters": tool_schema["input_schema"],
                    },
                }
        except Exception as e:
            print(
                f"⚠️  Error creating tool definition for {framework_tool.tool_name}: {e}"
            )
        return None

    def create_sub_agent_tool_definition_from_framework_agent(
        self, framework_agent
    ) -> Dict[str, Any]:
        """
        Create tool definition from FrameworkAgent (convert sub-agent to tool format)
        """
        # Sub-agents typically take a message parameter
        parameters = {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Task description or message to send to the sub-agent",
                }
            },
            "required": ["message"],
        }

        return {
            "type": "function",
            "function": {
                "name": f"{framework_agent.agent_name}_sub_agent",
                "description": framework_agent.description
                or f"Specialized agent for {framework_agent.agent_name}-related tasks",
                "parameters": parameters,
            },
        }

    def create_tool_definition_from_description(
        self, tool_name: str, tool_description: str, base_path: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """Create JSON Schema tool definition from tool name and description"""

        # First, try to get tool info from framework config
        if tool_name in self.framework_tools and base_path:
            framework_tool = self.framework_tools[tool_name]
            tool_schema = self.load_tool_schema_from_file(framework_tool, base_path)

            if tool_schema and "input_schema" in tool_schema:
                # Use the complete schema from framework config
                return {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_schema.get(
                            "description", framework_tool.description
                        ),
                        "parameters": tool_schema["input_schema"],
                    },
                }

        # Fallback: Extract parameters from usage section
        usage_pattern = r"Usage:\s*<tool_use>(.*?)</tool_use>"
        usage_match = re.search(usage_pattern, tool_description, re.DOTALL)

        parameters = {"type": "object", "properties": {}, "required": []}

        if usage_match:
            usage_content = usage_match.group(1)
            param_pattern = r"<(\w+)>(.*?)</\1>"
            param_matches = re.findall(param_pattern, usage_content, re.DOTALL)

            for param_name, param_desc in param_matches:
                if param_name in ["tool_name", "parameter"]:
                    continue

                # Parse parameter description for type and required info
                required_match = re.search(r"\(required", param_desc)
                type_match = re.search(r"type:\s*(\w+)", param_desc)

                param_type = type_match.group(1) if type_match else "string"

                # Clean up description
                clean_desc = re.sub(r"\s*\(.*?\)", "", param_desc).strip()

                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": clean_desc,
                }

                if required_match:
                    parameters["required"].append(param_name)

        # Extract description (first line)
        description_lines = tool_description.split("\n")
        description = (
            description_lines[0].strip() if description_lines else f"Tool: {tool_name}"
        )

        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": parameters,
            },
        }

    def create_sub_agent_tool_definition(
        self, agent_name: str, agent_description: str
    ) -> Dict[str, Any]:
        """
        Convert sub-agent definition to standard tool definition
        """
        # Extract description (first line)
        description_lines = agent_description.split("\n")
        description = (
            description_lines[0].strip()
            if description_lines
            else f"Sub-agent: {agent_name}"
        )

        # Sub-agents typically take a message parameter
        parameters = {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Task description or message to send to the sub-agent",
                }
            },
            "required": ["message"],
        }

        return {
            "type": "function",
            "function": {
                "name": f"{agent_name}_sub_agent",
                "description": description,
                "parameters": parameters,
            },
        }

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

    def _parse_xml_content_robust(self, xml_content: str):
        """Parse XML content using multiple strategies to handle malformed XML."""
        import html
        import re

        # Strategy 1: Try as-is
        try:
            return ET.fromstring(f"<root>{xml_content}</root>")
        except ET.ParseError as e:
            print(f"Initial XML parsing failed: {e}. Attempting recovery strategies...")

        # Strategy 2: Clean up common issues (unclosed tags, extra whitespace)
        try:
            cleaned_xml = xml_content.strip()

            # Fix potential unclosed tags by ensuring proper closing
            lines = cleaned_xml.split("\n")
            corrected_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for unclosed tags (opening tag without closing)
                tag_matches = re.findall(
                    r"<(\w+)(?:\s+[^>]*)?>([^<]*?)(?:</\1>|$)", line
                )
                if tag_matches:
                    # Line has proper tag structure
                    corrected_lines.append(line)
                else:
                    # Check if line has opening tag but no closing tag
                    opening_match = re.match(
                        r"<(\w+)(?:\s+[^>]*)?>\s*([^<]*)\s*$", line
                    )
                    if opening_match:
                        tag_name = opening_match.group(1)
                        content = opening_match.group(2)
                        corrected_lines.append(f"<{tag_name}>{content}</{tag_name}>")
                    else:
                        corrected_lines.append(line)

            cleaned_xml = "\n".join(corrected_lines)
            return ET.fromstring(f"<root>{cleaned_xml}</root>")

        except ET.ParseError:
            pass

        # Strategy 3: Escape HTML/XML content in parameter values
        try:

            def escape_param_content(match):
                param_name = match.group(1)
                param_content = match.group(2)

                # Escape HTML/XML content if it contains < >
                if "<" in param_content and ">" in param_content:
                    escaped_content = html.escape(param_content)
                    return f"<{param_name}>{escaped_content}</{param_name}>"
                return match.group(0)

            # Find and escape parameter content within parameters block
            escaped_xml = xml_content
            params_match = re.search(
                r"<parameter>(.*?)</parameter>", xml_content, re.DOTALL
            )
            if params_match:
                params_content = params_match.group(1)
                # Pattern to match individual parameter tags
                param_pattern = r"<(\w+)>(.*?)</\1>"
                escaped_params = re.sub(
                    param_pattern, escape_param_content, params_content, flags=re.DOTALL
                )
                escaped_xml = xml_content.replace(params_match.group(1), escaped_params)

            return ET.fromstring(f"<root>{escaped_xml}</root>")

        except ET.ParseError:
            pass

        # Strategy 4: Fallback - escape all content and selectively unescape XML tags
        try:
            escaped_content = html.escape(xml_content, quote=False)
            # Unescape the XML tags we need
            escaped_content = escaped_content.replace("&lt;", "<").replace("&gt;", ">")
            return ET.fromstring(f"<root>{escaped_content}</root>")
        except ET.ParseError:
            pass

        # Strategy 5: Extract content using regex and build minimal XML
        try:
            tool_name_match = re.search(
                r"<tool_name>\s*([^<]+)\s*</tool_name>",
                xml_content,
                re.IGNORECASE | re.DOTALL,
            )
            tool_name = (
                tool_name_match.group(1).strip() if tool_name_match else "unknown"
            )

            # Build minimal XML structure
            minimal_xml = f"<tool_name>{tool_name}</tool_name>"

            # Try to extract parameters
            params_match = re.search(
                r"<parameter>(.*?)</parameter>", xml_content, re.DOTALL | re.IGNORECASE
            )
            if params_match:
                params_content = params_match.group(1).strip()
                minimal_xml += f"<parameter>{params_content}</parameter>"

            return ET.fromstring(f"<root>{minimal_xml}</root>")
        except (ET.ParseError, AttributeError):
            pass

        # Final fallback: raise with detailed error
        raise ValueError(
            f"Unable to parse XML content after multiple strategies. Content preview: {xml_content[:200]}..."
        )

    def extract_tool_calls_from_xml(
        self, content: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Extract tool calls and sub-agent calls from XML format and convert to ChatCompletion format
        Supports: <tool_use>, <sub-agent>, <use_parallel_tool_calls>, <use_parallel_sub_agents>, <use_batch_agent>
        Returns (content_without_tool_calls, tool_calls)
        """
        tool_calls = []
        cleaned_content = self._restore_xml_closing_tags(content)

        # 1. Handle single tool_use calls - use more flexible pattern for formatted XML
        tool_use_pattern = r"<tool_use>(.*?)</tool_use>"

        def replace_tool_call(match):
            tool_xml_content = match.group(1).strip()

            try:
                # Use robust XML parsing like the agent implementation
                root = self._parse_xml_content_robust(tool_xml_content)

                # Get tool name
                tool_name_elem = root.find("tool_name")
                if tool_name_elem is None:
                    print(
                        f"⚠️  Missing tool_name in tool_use XML: {tool_xml_content[:100]}..."
                    )
                    return match.group(0)  # Return original if parsing fails

                tool_name = (tool_name_elem.text or "").strip()

                # Get parameters
                parameters = {}
                params_elem = root.find("parameter")
                if params_elem is not None:
                    for param in params_elem:
                        param_name = param.tag

                        # Handle both regular text and CDATA content
                        if param.text is not None:
                            param_value = param.text
                        else:
                            # Handle case where content is in CDATA or mixed content
                            param_value = "".join(param.itertext()) or ""

                        # Unescape HTML entities in parameter values
                        import html

                        param_value = html.unescape(param_value)
                        parameters[param_name] = param_value.strip()

                # Check if tool is registered in framework config
                if (
                    self.framework_config
                    and tool_name not in self.framework_tools
                    and tool_name not in ["StateSet", "StateGet", "StateAppend"]
                ):
                    print(
                        f"⚠️  Warning: Tool '{tool_name}' not found in framework config. Available tools: {list(self.framework_tools.keys())}"
                    )

                tool_call = {
                    "id": self._generate_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(parameters, ensure_ascii=False),
                    },
                }

                tool_calls.append(tool_call)
                return ""

            except Exception as e:
                print(f"⚠️  Error parsing tool_use XML: {e}")
                print(f"   Content: {tool_xml_content[:200]}...")
                # Fallback to regex parsing
                tool_name_match = re.search(
                    r"<tool_name>\s*([^<]+)\s*</tool_name>", tool_xml_content
                )
                if tool_name_match:
                    tool_name = tool_name_match.group(1).strip()
                    param_content = ""
                    param_match = re.search(
                        r"<parameter>(.*?)</parameter>", tool_xml_content, re.DOTALL
                    )
                    if param_match:
                        param_content = param_match.group(1).strip()

                    parameters = self.parse_xml_parameters(param_content)

                    # Check if tool is registered in framework config
                    if (
                        self.framework_config
                        and tool_name not in self.framework_tools
                        and tool_name not in ["StateSet", "StateGet", "StateAppend"]
                    ):
                        print(
                            f"⚠️  Warning: Tool '{tool_name}' not found in framework config. Available tools: {list(self.framework_tools.keys())}"
                        )

                    tool_call = {
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(parameters, ensure_ascii=False),
                        },
                    }

                    tool_calls.append(tool_call)
                    return ""

                return match.group(0)  # Return original if all parsing fails

        cleaned_content = re.sub(
            tool_use_pattern, replace_tool_call, cleaned_content, flags=re.DOTALL
        )

        # 2. Handle single sub-agent calls
        sub_agent_pattern = r"<sub-agent>\s*<agent_name>([^<]+)</agent_name>\s*<message>(.*?)</message>\s*</sub-agent>"

        def replace_sub_agent_call(match):
            agent_name = match.group(1).strip()
            message = match.group(2).strip()

            tool_call = {
                "id": self._generate_tool_call_id(),
                "type": "function",
                "function": {
                    "name": f"{agent_name}_sub_agent",
                    "arguments": json.dumps({"message": message}, ensure_ascii=False),
                },
            }

            tool_calls.append(tool_call)
            return ""

        cleaned_content = re.sub(
            sub_agent_pattern, replace_sub_agent_call, cleaned_content, flags=re.DOTALL
        )

        # 3. Handle parallel tool calls
        parallel_tools_pattern = (
            r"<use_parallel_tool_calls>(.*?)</use_parallel_tool_calls>"
        )

        def replace_parallel_tools(match):
            parallel_content = match.group(1)

            parallel_tool_pattern = r"<parallel_tool>(.*?)</parallel_tool>"
            parallel_tool_matches = re.findall(
                parallel_tool_pattern, parallel_content, re.DOTALL
            )

            for tool_xml_content in parallel_tool_matches:
                try:
                    # Use robust XML parsing like the agent implementation
                    root = self._parse_xml_content_robust(tool_xml_content.strip())

                    # Get tool name
                    tool_name_elem = root.find("tool_name")
                    if tool_name_elem is None:
                        print(
                            f"⚠️  Missing tool_name in parallel_tool XML: {tool_xml_content[:100]}..."
                        )
                        continue

                    tool_name = (tool_name_elem.text or "").strip()

                    # Get parameters
                    parameters = {}
                    params_elem = root.find("parameter")
                    if params_elem is not None:
                        for param in params_elem:
                            param_name = param.tag

                            # Handle both regular text and CDATA content
                            if param.text is not None:
                                param_value = param.text
                            else:
                                # Handle case where content is in CDATA or mixed content
                                param_value = "".join(param.itertext()) or ""

                            # Unescape HTML entities in parameter values
                            import html

                            param_value = html.unescape(param_value)
                            parameters[param_name] = param_value.strip()

                    # Check if tool is registered in framework config for parallel tools
                    if (
                        self.framework_config
                        and tool_name not in self.framework_tools
                        and tool_name not in ["StateSet", "StateGet", "StateAppend"]
                    ):
                        print(
                            f"⚠️  Warning: Parallel tool '{tool_name}' not found in framework config. Available tools: {list(self.framework_tools.keys())}"
                        )

                    tool_call = {
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(parameters, ensure_ascii=False),
                        },
                    }

                    tool_calls.append(tool_call)

                except Exception as e:
                    print(f"⚠️  Error parsing parallel_tool XML: {e}")
                    print(f"   Content: {tool_xml_content[:200]}...")
                    # Fallback to regex parsing
                    tool_name_match = re.search(
                        r"<tool_name>\s*([^<]+)\s*</tool_name>", tool_xml_content
                    )
                    param_match = re.search(
                        r"<parameter>(.*?)</parameter>", tool_xml_content, re.DOTALL
                    )

                    if tool_name_match:
                        tool_name = tool_name_match.group(1).strip()
                        param_content = (
                            param_match.group(1).strip() if param_match else ""
                        )
                        parameters = self.parse_xml_parameters(param_content)

                        # Check if tool is registered in framework config for fallback tools
                        if (
                            self.framework_config
                            and tool_name not in self.framework_tools
                            and tool_name not in ["StateSet", "StateGet", "StateAppend"]
                        ):
                            print(
                                f"⚠️  Warning: Tool '{tool_name}' (fallback) not found in framework config. Available tools: {list(self.framework_tools.keys())}"
                            )

                        tool_call = {
                            "id": self._generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(parameters, ensure_ascii=False),
                            },
                        }

                        tool_calls.append(tool_call)

            return ""

        cleaned_content = re.sub(
            parallel_tools_pattern,
            replace_parallel_tools,
            cleaned_content,
            flags=re.DOTALL,
        )

        # 4. Handle parallel sub-agents (can contain both agents and tools)
        parallel_sub_agents_pattern = (
            r"<use_parallel_sub_agents>(.*?)</use_parallel_sub_agents>"
        )

        def replace_parallel_sub_agents(match):
            parallel_content = match.group(1)

            # Extract parallel agents
            parallel_agent_pattern = r"<parallel_agent>\s*<agent_name>([^<]+)</agent_name>\s*<message>(.*?)</message>\s*</parallel_agent>"
            parallel_agent_matches = re.findall(
                parallel_agent_pattern, parallel_content, re.DOTALL
            )

            for agent_name, message in parallel_agent_matches:
                tool_call = {
                    "id": self._generate_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": f"{agent_name.strip()}_sub_agent",
                        "arguments": json.dumps(
                            {"message": message.strip()}, ensure_ascii=False
                        ),
                    },
                }

                tool_calls.append(tool_call)

            # Also extract parallel tools within this block
            parallel_tool_pattern = r"<parallel_tool>(.*?)</parallel_tool>"
            parallel_tool_matches = re.findall(
                parallel_tool_pattern, parallel_content, re.DOTALL
            )

            for tool_xml_content in parallel_tool_matches:
                try:
                    # Use robust XML parsing like the agent implementation
                    root = self._parse_xml_content_robust(tool_xml_content.strip())

                    # Get tool name
                    tool_name_elem = root.find("tool_name")
                    if tool_name_elem is None:
                        print(
                            f"⚠️  Missing tool_name in parallel_tool XML: {tool_xml_content[:100]}..."
                        )
                        continue

                    tool_name = (tool_name_elem.text or "").strip()

                    # Get parameters
                    parameters = {}
                    params_elem = root.find("parameter")
                    if params_elem is not None:
                        for param in params_elem:
                            param_name = param.tag

                            # Handle both regular text and CDATA content
                            if param.text is not None:
                                param_value = param.text
                            else:
                                # Handle case where content is in CDATA or mixed content
                                param_value = "".join(param.itertext()) or ""

                            # Unescape HTML entities in parameter values
                            import html

                            param_value = html.unescape(param_value)
                            parameters[param_name] = param_value.strip()

                    tool_call = {
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(parameters, ensure_ascii=False),
                        },
                    }

                    tool_calls.append(tool_call)

                except Exception as e:
                    print(
                        f"⚠️  Error parsing parallel_tool XML in sub_agents block: {e}"
                    )
                    print(f"   Content: {tool_xml_content[:200]}...")
                    # Fallback to regex parsing
                    tool_name_match = re.search(
                        r"<tool_name>\s*([^<]+)\s*</tool_name>", tool_xml_content
                    )
                    param_match = re.search(
                        r"<parameter>(.*?)</parameter>", tool_xml_content, re.DOTALL
                    )

                    if tool_name_match:
                        tool_name = tool_name_match.group(1).strip()
                        param_content = (
                            param_match.group(1).strip() if param_match else ""
                        )
                        parameters = self.parse_xml_parameters(param_content)

                        # Check if tool is registered in framework config for fallback tools
                        if (
                            self.framework_config
                            and tool_name not in self.framework_tools
                        ):
                            print(
                                f"⚠️  Warning: Tool '{tool_name}' (fallback) not found in framework config. Available tools: {list(self.framework_tools.keys())}"
                            )

                        tool_call = {
                            "id": self._generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(parameters, ensure_ascii=False),
                            },
                        }

                        tool_calls.append(tool_call)

            return ""

        cleaned_content = re.sub(
            parallel_sub_agents_pattern,
            replace_parallel_sub_agents,
            cleaned_content,
            flags=re.DOTALL,
        )

        # 5. Handle batch agent processing
        batch_agent_pattern = r"<use_batch_agent>\s*<agent_name>([^<]+)</agent_name>\s*<input_data_source>(.*?)</input_data_source>\s*<message>(.*?)</message>\s*</use_batch_agent>"

        def replace_batch_agent(match):
            agent_name = match.group(1).strip()
            input_data_source = match.group(2).strip()
            message = match.group(3).strip()

            # Parse input_data_source to extract file info
            file_pattern = r"<file_name>([^<]+)</file_name>"
            file_match = re.search(file_pattern, input_data_source)
            file_name = file_match.group(1).strip() if file_match else ""

            format_pattern = r"<format>([^<]+)</format>"
            format_match = re.search(format_pattern, input_data_source)
            data_format = format_match.group(1).strip() if format_match else "jsonl"

            tool_call = {
                "id": self._generate_tool_call_id(),
                "type": "function",
                "function": {
                    "name": f"{agent_name}_sub_agent",
                    "arguments": json.dumps(
                        {
                            "batch_mode": True,
                            "input_file": file_name,
                            "format": data_format,
                            "message_template": message,
                        },
                        ensure_ascii=False,
                    ),
                },
            }

            tool_calls.append(tool_call)
            return ""

        cleaned_content = re.sub(
            batch_agent_pattern, replace_batch_agent, cleaned_content, flags=re.DOTALL
        )

        return cleaned_content.strip(), tool_calls

    def parse_xml_parameters(self, param_content: str) -> Dict[str, Any]:
        """
        Parse XML parameter structure into a dictionary
        """
        parameters = {}

        if "<" in param_content and ">" in param_content:
            # Parse nested XML parameters
            param_pattern = r"<([^>]+)>(.*?)</\1>"
            param_matches = re.findall(param_pattern, param_content, re.DOTALL)

            for param_name, param_value in param_matches:
                parameters[param_name.strip()] = param_value.strip()
        else:
            # Handle simple text parameters (fallback)
            if param_content.strip():
                parameters["content"] = param_content.strip()

        return parameters

    def convert_tool_results_to_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert tool result messages from user role to tool role with proper ID mapping
        """
        converted_messages: List[Dict[str, Any]] = []
        pending_tool_calls: List[Dict[str, Optional[str]]] = []

        def _pop_next_call(tool_name: str) -> Dict[str, Optional[str]]:
            if pending_tool_calls:
                for idx, call_info in enumerate(pending_tool_calls):
                    call_name = call_info.get("name")
                    if call_name and call_name == tool_name:
                        return pending_tool_calls.pop(idx)
                return pending_tool_calls.pop(0)

            return {"id": self._generate_tool_call_id(), "name": tool_name}

        for message in messages:
            if message.get("role") == "assistant" and isinstance(
                message.get("tool_calls"), list
            ):
                for call in message["tool_calls"]:
                    if not isinstance(call, dict):
                        continue

                    call_id = call.get("id")
                    if not call_id:
                        call_id = self._generate_tool_call_id()
                        call["id"] = call_id

                    call_name = None
                    function_info = call.get("function")
                    if isinstance(function_info, dict):
                        call_name = function_info.get("name")

                    pending_tool_calls.append({"id": call_id, "name": call_name})

                converted_messages.append(message)
                continue

            if message.get("role") == "user" and message.get("content", "").startswith(
                "Tool execution results:"
            ):
                content = message["content"]

                tool_result_pattern = r"<tool_result>\s*<tool_name>([^<]+)</tool_name>\s*<result>(.*?)</result>\s*</tool_result>"
                tool_results = re.findall(tool_result_pattern, content, re.DOTALL)

                for tool_name, result_content in tool_results:
                    tool_name_clean = tool_name.strip()

                    if self.framework_config:
                        if (
                            tool_name_clean not in self.framework_tools
                            and tool_name_clean
                            not in ["StateSet", "StateGet", "StateAppend"]
                        ):
                            if tool_name_clean.endswith("_sub_agent"):
                                base_agent_name = tool_name_clean.replace(
                                    "_sub_agent", ""
                                )
                                agent_found = False
                                if self.framework_config.agents:
                                    for agent in self.framework_config.agents:
                                        if agent.agent_name == base_agent_name:
                                            agent_found = True
                                            break
                                if not agent_found:
                                    print(
                                        f"⚠️  Warning: Sub-agent '{base_agent_name}' not found in framework config. Tool result: '{tool_name_clean}'"
                                    )
                            else:
                                print(
                                    f"⚠️  Warning: Tool '{tool_name_clean}' not found in framework config (tool result). Available tools: {list(self.framework_tools.keys())}"
                                )

                    call_info = _pop_next_call(tool_name_clean)
                    tool_call_id = call_info.get("id") or self._generate_tool_call_id()
                    call_name_recorded = call_info.get("name")

                    if (
                        call_name_recorded
                        and call_name_recorded != tool_name_clean
                        and self.framework_config
                    ):
                        print(
                            f"⚠️  Warning: Tool result name '{tool_name_clean}' does not match pending call '{call_name_recorded}'. Using pending call ID {tool_call_id}."
                        )

                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name_clean,
                        "content": result_content.strip(),
                    }
                    converted_messages.append(tool_message)

                if not tool_results and content.strip():
                    tool_name = "unknown_tool"

                    if self.framework_config:
                        print(
                            f"⚠️  Warning: Using fallback tool name '{tool_name}' for unstructured tool result. Consider using proper <tool_result> format."
                        )

                    call_info = _pop_next_call(tool_name)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call_info.get("id")
                        or self._generate_tool_call_id(),
                        "name": tool_name,
                        "content": content.replace(
                            "Tool execution results:", ""
                        ).strip(),
                    }
                    converted_messages.append(tool_message)

                continue

            converted_messages.append(message)

        return converted_messages

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
        tools = []

        for message in input_data:
            if message.get("role") == "system":
                # Get base path for framework config if available
                base_path = None
                if self.framework_config:
                    base_path = (
                        Path(self.framework_config_path).parent
                        if hasattr(self, "framework_config_path")
                        else None
                    )

                cleaned_content, extracted_tools = (
                    self.extract_tool_definitions_from_schema(
                        message["content"], base_path, agent_name
                    )
                )
                messages.append({"role": "system", "content": cleaned_content})
                tools.extend(extracted_tools)
            elif message.get("role") == "assistant":
                content = message.get("content", "")
                assistant_content, tool_calls = self.extract_tool_calls_from_xml(
                    content
                )
                message["content"] = assistant_content
                if tool_calls:
                    message["tool_calls"] = tool_calls
                messages.append(message)
            else:
                messages.append(message)

        # Convert tool results in messages
        messages = self.convert_tool_results_to_messages(messages)

        # Extract tool calls from assistant output
        assistant_content = ""
        tool_calls = []

        if output_data and output_data.get("role") == "assistant":
            content = output_data.get("content", "")
            assistant_content, tool_calls = self.extract_tool_calls_from_xml(content)

        # Build request
        request = {
            "model": span.get("model", "nex"),
            "messages": messages,
        }

        if tools:
            request["tools"] = tools

        # Build response
        response_message = {
            "role": "assistant",
            "content": assistant_content if assistant_content else None,
        }

        if tool_calls:
            response_message["tool_calls"] = tool_calls

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
            "model": span.get("model", "nex"),
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": "stop" if not tool_calls else "tool_calls",
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

        # Extract tools from request (if any)
        tools = request.get("tools", [])

        # Extract response message from response
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice:
                response_message = choice["message"].copy()

                # Ensure the response message has the correct structure
                if "role" not in response_message:
                    response_message["role"] = "assistant"

                # Convert tool_calls arguments from JSON strings to objects if needed
                if "tool_calls" in response_message and response_message["tool_calls"]:
                    for tool_call in response_message["tool_calls"]:
                        if (
                            "function" in tool_call
                            and "arguments" in tool_call["function"]
                        ):
                            # If arguments is a string, parse it as JSON
                            if isinstance(tool_call["function"]["arguments"], str):
                                try:
                                    tool_call["function"]["arguments"] = json.loads(
                                        tool_call["function"]["arguments"], strict=False
                                    )
                                except json.JSONDecodeError:
                                    # If parsing fails, set to empty dict
                                    tool_call["function"]["arguments"] = {}

                messages.append(response_message)

        # Preserve metadata from original chatcompletion
        result = {"messages": messages, "tools": tools}

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
    parser.add_argument("-c", "--config", help="Path to framework config YAML file")
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
        converter = SpansToChatCompletionConverter(framework_config_path=args.config)

        if args.filter_only:
            # Only filter spans, don't convert
            output_file = converter.filter_spans_file(args.input, args.output)

            if args.verbose:
                print("\n📋 Filter Details:")
                print(
                    f"  - Framework config: {'✅ Loaded' if converter.framework_config else '❌ Not provided'}"
                )
                print(
                    f"  - Filtered to keep only last OpenAI generation per span group"
                )
        else:
            # Regular conversion
            output_file = converter.convert_spans_file(args.input, args.output)

            if args.verbose:
                print("\n📋 Conversion Details:")
                print(
                    f"  - Framework config: {'✅ Loaded' if converter.framework_config else '❌ Not provided'}"
                )
                print(
                    f"  - Tool definitions extracted from system prompts and framework config"
                )
                print(f"  - XML tool calls converted to standard format")
                print(f"  - Tool results converted from user to tool messages")
                print(f"  - Only OpenAI generation spans were processed")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
