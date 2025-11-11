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
MCP Tools Preloader
Preload MCP server tool definitions and build agent→tools mapping cache
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set

import yaml


class MCPPreloader:
    """MCP Tool Preloader"""

    def __init__(self, base_dir: Path, cache_manager=None):
        self.base_dir = base_dir
        self.cache_manager = cache_manager
        self.mcp_cache = {}  # url -> tools
        self.agent_tools_cache = {}  # agent_name -> [tool_names]

    def preload_all_configs(
        self, verbose: bool = False, use_cache: bool = True
    ) -> Dict:
        """Scan all framework_config.yaml files and preload MCP tools"""
        # Try to load from cache
        if use_cache and self.cache_manager:
            cached_mcp = self.cache_manager.get_mcp_cache()
            cached_agent_tools = self.cache_manager.get_agent_tools_cache()

            if cached_mcp and cached_agent_tools:
                if verbose:
                    print(
                        f"✅ Loaded from cache: {len(cached_mcp)} MCP servers, {len(cached_agent_tools)} Agents"
                    )
                self.mcp_cache = cached_mcp
                self.agent_tools_cache = cached_agent_tools
                return {
                    "mcp_cache": self.mcp_cache,
                    "agent_tools_cache": self.agent_tools_cache,
                }

        if verbose:
            print("🔄 Preloading MCP tool definitions...")

        # Find all framework_config.yaml files
        config_files = []
        for pattern in ["created_subagents", "created_frameworks", "subagents"]:
            search_dir = self.base_dir / pattern
            if search_dir.exists():
                config_files.extend(search_dir.rglob("framework_config.yaml"))

        # Add MetaAgent config
        meta_config = self.base_dir / "framework_config.yaml"
        if meta_config.exists():
            config_files.append(meta_config)

        if verbose:
            print(f"   Found {len(config_files)} config files")

        # Process each config file
        for config_file in config_files:
            self._process_config(config_file, verbose)

        if verbose:
            print(
                f"✅ Preload complete: {len(self.mcp_cache)} MCP servers, {len(self.agent_tools_cache)} Agents"
            )

        # Save to cache
        if self.cache_manager:
            self.cache_manager.set_mcp_cache(self.mcp_cache)
            self.cache_manager.set_agent_tools_cache(self.agent_tools_cache)
            if verbose:
                print("   Saved to cache")

        return {
            "mcp_cache": self.mcp_cache,
            "agent_tools_cache": self.agent_tools_cache,
        }

    def _process_config(self, config_file: Path, verbose: bool = False):
        """Process a single framework_config.yaml"""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not config:
                return

            # Preload MCP servers
            mcp_servers = config.get("mcp_servers", [])
            for mcp_server in mcp_servers:
                if not isinstance(mcp_server, dict):
                    continue

                url = mcp_server.get("url")
                if not url or url in self.mcp_cache:
                    continue

                # Fetch MCP tools
                tools = self._fetch_mcp_tools(url, verbose)
                if tools:
                    self.mcp_cache[url] = tools

            # Build agent→tools mapping
            agents = config.get("agents", [])
            for agent in agents:
                if not isinstance(agent, dict):
                    continue

                agent_name = agent.get("agent_name") or agent.get("name")
                if not agent_name:
                    continue

                # Get agent's tool list
                tool_names = set()

                # Get from tools field
                tools_list = agent.get("tools", [])
                for tool in tools_list:
                    if isinstance(tool, str):
                        tool_names.add(tool)
                    elif isinstance(tool, dict):
                        tool_name = tool.get("name")
                        if tool_name:
                            tool_names.add(tool_name)

                # Get from mcp_servers
                agent_mcp_servers = agent.get("mcp_servers", [])
                for mcp_url in agent_mcp_servers:
                    if mcp_url in self.mcp_cache:
                        mcp_tools = self.mcp_cache[mcp_url]
                        tool_names.update(
                            t.get("name") for t in mcp_tools if t.get("name")
                        )

                if tool_names:
                    self.agent_tools_cache[agent_name] = list(tool_names)

        except Exception as e:
            if verbose:
                print(f"   ⚠️  Failed to process config file {config_file.name}: {e}")

    def _fetch_mcp_tools(self, url: str, verbose: bool = False) -> List[Dict]:
        """Fetch tool definitions from MCP server"""
        try:
            cmd = ["mcp", "dev", url, "--method", "tools/list"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                tools_data = json.loads(result.stdout)
                tools = tools_data.get("tools", [])
                if verbose and tools:
                    print(f"   ✅ MCP {url}: {len(tools)} tools")
                return tools
        except Exception as e:
            if verbose:
                print(f"   ⚠️  MCP fetch failed {url}: {e}")

        return []
