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
Framework Config Schema using Pydantic

This module defines the data models for framework_config format
using Pydantic for type validation, default values, and documentation.

Based on the simplified framework configuration structure.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, validator


class FrameworkLLMConfig(BaseModel):
    """LLM configuration for framework_config"""

    llm_name: str = Field(..., description="该配置名称（作为唯一标识）")
    provider: str = Field(
        ...,
        description="LLM提供商（openai, anthropic, huggingface, ollama, azure, google等）",
    )
    model: str = Field(
        ...,
        description="模型名称",
        examples=["gpt-4", "claude-3", "llama-2-70b", "gemini-pro"],
    )
    api_key: Optional[str] = Field(None, description="API密钥（支持环境变量引用）")
    base_url: Optional[str] = Field(None, description="自定义基础URL")
    max_tokens: int = Field(4000, description="最大Token数", ge=1, le=200000)
    temperature: float = Field(0.7, description="温度参数", ge=0.0, le=2.0)
    timeout: int = Field(60, description="请求超时时间（秒）", ge=1, le=300)


class FrameworkAgent(BaseModel):
    """Agent definition for framework_config"""

    agent_name: str = Field(..., description="Agent名称（作为唯一标识）")
    description: Optional[str] = Field(None, description="Agent描述（可选）")
    sysprompt_path: str = Field(..., description="系统提示文件路径")
    llm_config: str = Field(..., description="Agent使用的LLM配置名称")
    tools: List[str] = Field(
        default_factory=list, description="可用工具列表（工具名称引用）"
    )
    mcp_servers: Optional[List[str]] = Field(
        None, description="Agent使用的MCP服务器列表"
    )


class FrameworkTool(BaseModel):
    """Tool definition for framework_config"""

    tool_name: str = Field(..., description="工具名称（作为唯一标识）")
    description: str = Field(..., description="工具用途描述")
    config_path: str = Field(
        ..., description="工具YAML配置文件路径（相对于当前配置文件）"
    )
    binding: str = Field(
        ..., description="Python函数绑定信息（module.path:function_name）"
    )


class FrameworkWorkflowNode(BaseModel):
    """Workflow node for framework_config"""

    id: str = Field(..., description="节点ID（必须唯一）")
    type: Literal["agent", "tool", "mcp"] = Field(..., description="节点类型")


class FrameworkWorkflowEdge(BaseModel):
    """Workflow edge for framework_config"""

    from_: str = Field(..., alias="from", description="起始节点ID")
    to: str = Field(..., description="目标节点ID")


class FrameworkWorkflow(BaseModel):
    """Workflow definition for framework_config"""

    name: str = Field(..., description="工作流名称")
    description: Optional[str] = Field(None, description="工作流描述")
    nodes: List[FrameworkWorkflowNode] = Field(
        default_factory=list, description="工作流节点定义"
    )
    edges: List[FrameworkWorkflowEdge] = Field(
        default_factory=list, description="工作流边定义"
    )


class FrameworkMCPServer(BaseModel):
    """MCP Server definition for framework_config"""

    name: str = Field(..., description="MCP服务器名称")
    type: str = Field(..., description="MCP服务器类型（sse等）")
    url: str = Field(..., description="MCP服务器URL")
    timeout: int = Field(60, description="超时时间（秒）")
    description: Optional[str] = Field(None, description="服务器描述")


class FrameworkConfig(BaseModel):
    """Complete framework_config schema - simplified version"""

    agents: List[FrameworkAgent] = Field(..., description="Agent定义列表")
    llm_configs: List[FrameworkLLMConfig] = Field(..., description="LLM配置列表")
    tools: Optional[List[FrameworkTool]] = Field(None, description="工具配置列表")
    workflow: Optional[FrameworkWorkflow] = Field(None, description="工作流配置")
    mcp_servers: Optional[List[FrameworkMCPServer]] = Field(
        None, description="MCP服务器配置列表"
    )
    framework_name: Optional[str] = Field(None, description="框架名称")
    framework_entrance_agent: Optional[str] = Field(None, description="框架入口agent")

    class Config:
        extra = "allow"  # Changed from "forbid" to allow future extensions
        allow_population_by_field_name = True

    @field_validator("agents")
    def validate_agents_not_empty(cls, v):
        if not v:
            raise ValueError("至少需要一个Agent定义")
        return v

    @field_validator("llm_configs")
    def validate_llm_configs_not_empty(cls, v):
        if not v:
            raise ValueError("至少需要一个LLM配置")
        return v
