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
Agent package for multi-agent orchestration within the Gyrfalcon pipeline.

Provides shared entry points for agent base classes and default agents.
"""

from .base import Agent, AgentContext, AgentOrchestrator, AgentOutput
from .file_augmentation_agent import FileAugmentationAgent
from .file_requirement_agent import FileRequirementAgent
from .file_system_agent import FileSystemAgent
from .fuzzifier_agent import FuzzifierAgent
from .query_synthesis_agent import QuerySynthesisAgent
from .rewrite_agent import RewriteAgent
from .router_agent import RouterAgent
from .url_extraction_agent import URLExtractionAgent
from .url_processing_agent import URLProcessingAgent
from .url_query_rewrite_agent import URLQueryRewriteAgent
from .url_repair_agent import URLRepairAgent
from .url_validator_agent import URLValidatorAgent
from .web_research_agent import WebResearchAgent

__all__ = [
    "Agent",
    "AgentContext",
    "AgentOutput",
    "AgentOrchestrator",
    "RewriteAgent",
    "QuerySynthesisAgent",
    "FileRequirementAgent",
    "FileSystemAgent",
    "FileAugmentationAgent",
    "RouterAgent",
    "WebResearchAgent",
    "FuzzifierAgent",
    "URLExtractionAgent",
    "URLValidatorAgent",
    "URLRepairAgent",
    "URLQueryRewriteAgent",
    "URLProcessingAgent",
]
