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
Problem Type Tree Module for Gyrfalcon v3

This module manages hierarchical problem type taxonomies for frameworks,
replacing the knowledge graph approach with a simpler tree structure.
"""

from .tree_manager import (
    ProblemTypeNode,
    ProblemTypeTree,
    ProblemTypeTreeManager,
    TagTrace,
)
from .visualizer import ProblemTypeTreeVisualizer, create_visualization_for_framework

__all__ = [
    "ProblemTypeNode",
    "ProblemTypeTree",
    "ProblemTypeTreeManager",
    "TagTrace",
    "ProblemTypeTreeVisualizer",
    "create_visualization_for_framework",
]
