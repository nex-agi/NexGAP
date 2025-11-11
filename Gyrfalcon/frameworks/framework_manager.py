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
Framework Configuration Manager for Gyrfalcon v2 Pipeline

This module manages the configuration and data for different agent frameworks,
including reading config files, persona data, and related metadata.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FrameworkPersona:
    """Represents a user persona for a framework"""

    persona: str
    persona_chinese: str = ""  # 中文版本
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "FrameworkPersona":
        """Create persona from dictionary"""
        persona = data.get("persona", "")
        persona_chinese = data.get("persona_chinese", "")
        metadata = {
            k: v for k, v in data.items() if k not in ["persona", "persona_chinese"]
        }
        return cls(persona=persona, persona_chinese=persona_chinese, metadata=metadata)

    def get_persona(self, language: str = "english") -> str:
        """Get persona description in specified language"""
        if language.lower() == "chinese" and self.persona_chinese:
            return self.persona_chinese
        return self.persona


@dataclass
class FrameworkConfig:
    """Complete configuration for an agent framework"""

    name: str
    description: str = ""
    subagents: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    mcp: List[Dict[str, Any]] = field(default_factory=list)
    personas: List[FrameworkPersona] = field(default_factory=list)
    knowledge_graph_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict) -> "FrameworkConfig":
        """Create framework config from dictionary"""
        # If 'metadata' field exists in data, use it directly; otherwise collect extra fields
        if "metadata" in data:
            metadata = data.get("metadata", {})
        else:
            metadata = {
                k: v
                for k, v in data.items()
                if k
                not in [
                    "description",
                    "subagents",
                    "tools",
                    "mcp",
                    "knowledge_graph_config",
                    "metadata",
                ]
            }

        return cls(
            name=name,
            description=data.get("description", ""),
            subagents=data.get("subagents", []),
            tools=data.get("tools", []),
            mcp=data.get("mcp", []),
            knowledge_graph_config=data.get("knowledge_graph_config", {}),
            metadata=metadata,
        )


class FrameworkConfigManager:
    """
    Manages configuration for all agent frameworks.
    Handles loading and parsing of config files, personas, and related metadata.
    """

    def __init__(self, frameworks_dir: str):
        self.frameworks_dir = Path(frameworks_dir)
        self.frameworks: Dict[str, FrameworkConfig] = {}
        self._loaded = False

    def load_all_frameworks(self) -> Dict[str, FrameworkConfig]:
        """
        Load all framework configurations from the frameworks directory.
        """
        if not self.frameworks_dir.exists():
            raise FileNotFoundError(
                f"Frameworks directory not found: {self.frameworks_dir}"
            )

        self.frameworks.clear()

        # Scan for framework subdirectories
        for framework_path in self.frameworks_dir.iterdir():
            if (
                framework_path.is_dir()
                and not framework_path.name.startswith(".")
                and framework_path.name != "__pycache__"
            ):
                try:
                    framework_config = self._load_framework(framework_path)
                    self.frameworks[framework_config.name] = framework_config
                    logger.info(f"Loaded framework: {framework_config.name}")
                except Exception as e:
                    logger.error(f"Failed to load framework {framework_path.name}: {e}")

        self._loaded = True
        return self.frameworks

    def load_framework(self, framework_name: str) -> FrameworkConfig:
        """Load a single framework configuration on demand."""
        if framework_name in self.frameworks:
            return self.frameworks[framework_name]

        framework_path = self.frameworks_dir / framework_name
        if not framework_path.exists() or not framework_path.is_dir():
            raise FileNotFoundError(f"Framework directory not found: {framework_path}")

        framework_config = self._load_framework(framework_path)
        self.frameworks[framework_config.name] = framework_config
        logger.info(f"Loaded framework: {framework_config.name}")
        return framework_config

    def _load_framework(self, framework_path: Path) -> FrameworkConfig:
        """
        Load a single framework configuration from its directory.
        """
        framework_name = framework_path.name

        # Load main config file
        config_file = framework_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        framework_config = FrameworkConfig.from_dict(framework_name, config_data)

        # Load personas
        persona_file = framework_path / "persona.jsonl"
        if persona_file.exists():
            framework_config.personas = self._load_personas(persona_file)
        else:
            logger.warning(f"No persona file found for framework: {framework_name}")

        return framework_config

    def _load_personas(self, persona_file: Path) -> List[FrameworkPersona]:
        """
        Load personas from a JSONL file.
        """
        personas = []
        try:
            with open(persona_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        persona_data = json.loads(line)
                        persona = FrameworkPersona.from_dict(persona_data)
                        personas.append(persona)
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Invalid JSON in {persona_file} line {line_num}: {e}"
                        )
        except Exception as e:
            logger.error(f"Error reading persona file {persona_file}: {e}")

        return personas

    def get_framework(self, name: str) -> Optional[FrameworkConfig]:
        """
        Get a specific framework configuration by name.
        """
        if name in self.frameworks:
            return self.frameworks[name]

        try:
            return self.load_framework(name)
        except FileNotFoundError:
            logger.error(f"Framework '{name}' not found in {self.frameworks_dir}")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load framework '{name}': {exc}")
            return None

    def get_all_frameworks(self) -> Dict[str, FrameworkConfig]:
        """
        Get all loaded framework configurations.
        """
        if not self._loaded:
            self.load_all_frameworks()
        return self.frameworks.copy()

    def get_framework_names(self) -> List[str]:
        """
        Get list of all framework names.
        """
        if not self._loaded:
            self.load_all_frameworks()
        return list(self.frameworks.keys())

    def validate_framework_config(self, framework_name: str) -> Dict[str, Any]:
        """
        Validate a framework configuration and return validation results.
        """
        framework = self.get_framework(framework_name)
        if not framework:
            return {"valid": False, "error": f"Framework '{framework_name}' not found"}

        validation_result = {"valid": True, "warnings": [], "stats": {}}

        # Check for required components
        if not framework.personas:
            validation_result["warnings"].append("No personas defined")

        # Collect statistics
        validation_result["stats"] = {
            "num_personas": len(framework.personas),
            "num_subagents": len(framework.subagents),
            "num_tools": len(framework.tools),
            "num_mcp": len(framework.mcp),
        }

        # Check for potential issues
        if len(framework.personas) == 0:
            validation_result["warnings"].append(
                "Framework has no personas - queries may be generic"
            )

        return validation_result

    def create_framework_template(
        self, framework_name: str, output_dir: Optional[str] = None
    ) -> Path:
        """
        Create a template directory structure for a new framework.
        """
        if output_dir is None:
            output_dir = self.frameworks_dir

        framework_path = Path(output_dir) / framework_name
        framework_path.mkdir(parents=True, exist_ok=True)

        # Create config.json template
        config_template = {
            "description": f"Configuration for {framework_name} agent framework",
            "subagents": [
                {
                    "name": "example_subagent",
                    "description": "An example subagent",
                    "capabilities": ["reasoning", "planning"],
                }
            ],
            "tools": [
                {
                    "name": "example_tool",
                    "description": "An example tool",
                    "parameters": {},
                }
            ],
            "mcp": [],
            "knowledge_graph_config": {"max_depth": 10, "similarity_threshold": 0.85},
        }

        config_file = framework_path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_template, f, indent=2, ensure_ascii=False)

        # Create persona.jsonl template
        persona_template = [
            {"persona": "A software engineer working on distributed systems"},
            {"persona": "A data scientist analyzing customer behavior"},
            {"persona": "A project manager coordinating team activities"},
        ]

        persona_file = framework_path / "persona.jsonl"
        with open(persona_file, "w", encoding="utf-8") as f:
            for persona in persona_template:
                f.write(json.dumps(persona, ensure_ascii=False) + "\n")

        return framework_path

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all loaded frameworks.
        """
        if not self._loaded:
            self.load_all_frameworks()

        total_personas = sum(len(fw.personas) for fw in self.frameworks.values())

        return {
            "total_frameworks": len(self.frameworks),
            "framework_names": list(self.frameworks.keys()),
            "total_personas": total_personas,
            "frameworks_detail": {
                name: {
                    "personas": len(fw.personas),
                    "subagents": len(fw.subagents),
                    "tools": len(fw.tools),
                }
                for name, fw in self.frameworks.items()
            },
        }
