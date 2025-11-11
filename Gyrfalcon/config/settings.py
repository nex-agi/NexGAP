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
Global configuration for Gyrfalcon v3 pipeline
"""
import os

# LLM Configuration
# Reads from environment variables (recommended)
# Supports both OPENAI_* and LLM_* prefixes for compatibility
LLM_CONFIG = {
    "base_url": os.getenv("OPENAI_BASE_URL")
    or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    "api_key": os.getenv("OPENAI_API_KEY")
    or os.getenv("LLM_API_KEY", "<YOUR_API_KEY>"),
    "model_name": os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL", "gpt-4o"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "max_tokens": int(
        os.getenv("LLM_MAX_TOKENS", "16384")
    ),  # Increased to accommodate reasoning tokens + content generation
    "timeout": float(
        os.getenv("LLM_TIMEOUT", "600.0")
    ),  # Timeout in seconds (default: 10 minutes)
    "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),  # Maximum retry attempts
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    "num_queries_per_run": 3,  # Number of different difficulty queries to generate
    "tag_trace_max_depth": 10,  # Maximum depth for tag trace generation
    "framework_data_dir": "frameworks",
    "knowledge_graph_dir": "knowledge_graph",
    "output_dir": "output",
}

# Knowledge Graph Configuration
KNOWLEDGE_GRAPH_CONFIG = {
    "similarity_threshold": 0.85,  # Threshold for considering tags similar
    "edge_weight_decay": 0.1,  # Decay factor for edge weights
    "max_connections_per_tag": 50,  # Maximum connections per tag
}
