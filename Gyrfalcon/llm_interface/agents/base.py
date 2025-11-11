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
Base abstractions for multi-agent orchestration in the Gyrfalcon pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """
    Shared context passed between agents.
    Acts as a lightweight state container with timing and error tracking.
    """

    data: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, values: Dict[str, Any]) -> None:
        if not values:
            return
        self.data.update(values)

    def add_timing(self, key: str, value: float) -> None:
        self.timings[key] = value

    def append_error(self, error: str) -> None:
        if error:
            self.errors.append(error)


@dataclass
class AgentOutput:
    """
    Standardized output structure for agents.
    """

    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class Agent(ABC):
    """
    Base Agent class. Custom agents should inherit and implement `run`.
    """

    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, context: AgentContext) -> AgentOutput:
        """Execute agent logic with shared context."""


class AgentOrchestrator:
    """
    Simple orchestrator that executes a sequence of agents.
    Stops execution when an agent reports failure.
    """

    def __init__(self, agents: List[Agent]):
        self.agents = agents

    def execute(self, context: AgentContext) -> AgentContext:
        for agent in self.agents:
            logger.debug("Running agent: %s", agent.name)
            try:
                output = agent.run(context)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent %s raised an exception: %s", agent.name, exc)
                context.append_error(f"{agent.name}: {exc}")
                break

            # Merge timings first so downstream agents can access them
            for key, value in output.timings.items():
                context.add_timing(key, value)

            if not output.success:
                if output.errors:
                    for error in output.errors:
                        context.append_error(f"{agent.name}: {error}")
                else:
                    context.append_error(
                        f"{agent.name}: execution failed without details"
                    )
                break

            context.update(output.data)

        return context
