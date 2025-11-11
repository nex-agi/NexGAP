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
NexAU Config Schema using Pydantic

This module defines the data models for nexau_config format
using Pydantic for type validation, default values, and documentation.

Based on the NexAU framework configuration structure.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class NexauLLMConfig(BaseModel):
    """LLM configuration for nexau_config"""

    model: Optional[str] = Field(
        None, description="模型名称，默认从环境变量LLM_MODEL获取"
    )
    base_url: Optional[str] = Field(
        None, description="API基础URL，默认从LLM_BASE_URL获取"
    )
    api_key: Optional[str] = Field(None, description="API密钥，默认从LLM_API_KEY获取")

    # 生成参数
    temperature: float = Field(0.7, description="温度参数", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="最大生成tokens", ge=1)
    top_p: Optional[float] = Field(None, description="Top-p采样参数", ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(
        None, description="频率惩罚", ge=-2.0, le=2.0
    )
    presence_penalty: Optional[float] = Field(
        None, description="存在惩罚", ge=-2.0, le=2.0
    )

    # 请求配置
    timeout: Optional[float] = Field(None, description="请求超时秒数", gt=0)
    max_retries: int = Field(3, description="最大重试次数", ge=0)
    debug: bool = Field(False, description="调试模式")


class NexauTool(BaseModel):
    """Tool configuration for nexau_config"""

    name: str = Field(..., description="工具名称")
    yaml_path: str = Field(..., description="工具YAML配置文件路径")
    binding: str = Field(
        ..., description="工具绑定路径，格式：module.path:function_name"
    )


class NexauSubAgent(BaseModel):
    """Sub-agent configuration for nexau_config"""

    name: str = Field(..., description="子Agent名称")
    config_path: str = Field(..., description="子Agent配置文件路径")


class NexauConfig(BaseModel):
    """Complete nexau_config schema"""

    # 基础配置
    name: str = Field(..., description="Agent名称，作为唯一标识")
    max_context: int = Field(100000, description="最大上下文长度", ge=1)
    max_running_subagents: int = Field(5, description="最大并发子Agent数量", ge=1)

    # 系统提示配置
    system_prompt: str = Field(..., description="系统提示内容，支持Jinja2模板变量")
    system_prompt_type: Literal["string", "file", "jinja"] = Field(
        "string", description="提示类型"
    )

    # LLM配置（可选）
    llm_config: Optional[NexauLLMConfig] = Field(None, description="LLM配置对象")

    # 工具和子Agent配置（可选）
    tools: List[NexauTool] = Field(default_factory=list, description="工具配置列表")
    sub_agents: List[NexauSubAgent] = Field(
        default_factory=list, description="子Agent配置列表"
    )

    class Config:
        extra = "forbid"
        allow_population_by_field_name = True

    @field_validator("name")
    def validate_name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Agent名称不能为空")
        return v

    @field_validator("system_prompt")
    def validate_system_prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("系统提示不能为空")
        return v

    @field_validator("tools")
    def validate_tool_names_unique(cls, v):
        names = [tool.name for tool in v]
        if len(names) != len(set(names)):
            raise ValueError("工具名称必须唯一")
        return v

    @field_validator("sub_agents")
    def validate_subagent_names_unique(cls, v):
        names = [agent.name for agent in v]
        if len(names) != len(set(names)):
            raise ValueError("子Agent名称必须唯一")
        return v
