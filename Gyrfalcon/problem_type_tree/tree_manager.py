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
Problem Type Tree Manager

Manages hierarchical tree structures of problem types for frameworks.
Supports bilingual (English/Chinese) problem type definitions and
provides functionality for intelligent trace generation with sampling statistics
and dynamic tag expansion.

Includes file-based locking for multi-process safety.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .file_lock import tag_tree_lock
from .sampling_manager import NewTagGenerator, PathSamplingStats

logger = logging.getLogger(__name__)


@dataclass
class ProblemTypeNode:
    """
    Represents a node in the problem type tree.
    Each node has bilingual labels (en/zh) and optional children.
    """

    id: str
    en: str  # English label
    zh: str  # Chinese label
    children: List["ProblemTypeNode"] = field(default_factory=list)
    parent: Optional["ProblemTypeNode"] = None

    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0

    def get_label(self, language: str = "english") -> str:
        """Get label in specified language"""
        return self.zh if language.lower() in ["chinese", "zh", "zh-cn"] else self.en

    def get_all_paths_to_leaves(self) -> List[List["ProblemTypeNode"]]:
        """Get all paths from this node to all leaf nodes"""
        if self.is_leaf():
            return [[self]]

        all_paths = []
        for child in self.children:
            child_paths = child.get_all_paths_to_leaves()
            for path in child_paths:
                all_paths.append([self] + path)

        return all_paths

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        result = {"id": self.id, "en": self.en, "zh": self.zh}
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], parent: Optional["ProblemTypeNode"] = None
    ) -> "ProblemTypeNode":
        """Create node from dictionary"""
        node = cls(id=data["id"], en=data["en"], zh=data["zh"], parent=parent)

        if "children" in data:
            for child_data in data["children"]:
                child_node = cls.from_dict(child_data, parent=node)
                node.children.append(child_node)

        return node


@dataclass
class TagTrace:
    """
    Represents a trace (path) through the problem type tree.
    Contains both node objects and string representations.
    """

    nodes: List[ProblemTypeNode]
    language: str = "english"

    def get_labels(self) -> List[str]:
        """Get all labels in the trace"""
        return [node.get_label(self.language) for node in self.nodes]

    def get_label_string(self, separator: str = " → ") -> str:
        """Get formatted string of all labels"""
        return separator.join(self.get_labels())

    def get_leaf_node(self) -> ProblemTypeNode:
        """Get the leaf node (last node in trace)"""
        return self.nodes[-1] if self.nodes else None

    def get_random_intermediate_node(self) -> Optional[ProblemTypeNode]:
        """Get a random intermediate node (not root, not leaf)"""
        if len(self.nodes) <= 2:
            return None
        # Exclude root (index 0) and leaf (index -1)
        intermediate_nodes = self.nodes[1:-1]
        return random.choice(intermediate_nodes) if intermediate_nodes else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary"""
        return {
            "node_ids": [node.id for node in self.nodes],
            "labels": self.get_labels(),
            "language": self.language,
            "path_string": self.get_label_string(),
        }


class ProblemTypeTree:
    """
    Manages a complete problem type tree for a framework.
    """

    def __init__(self, framework_name: str):
        self.framework_name = framework_name
        self.root: Optional[ProblemTypeNode] = None
        self.node_index: Dict[str, ProblemTypeNode] = {}

    def load_from_file(self, filepath: str):
        """Load tree structure from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("framework") != self.framework_name:
            logger.warning(
                f"Framework mismatch: expected {self.framework_name}, "
                f"got {data.get('framework')}"
            )

        # Build tree from data
        self.root = ProblemTypeNode.from_dict(data["tree"])

        # Build index for quick node lookup
        self._build_node_index(self.root)

        logger.info(
            f"Loaded problem type tree for {self.framework_name}: "
            f"{len(self.node_index)} nodes"
        )

    def _build_node_index(self, node: ProblemTypeNode):
        """Build index of all nodes for quick lookup"""
        self.node_index[node.id] = node
        for child in node.children:
            self._build_node_index(child)

    def get_all_paths(self) -> List[List[ProblemTypeNode]]:
        """Get all paths from root to leaves"""
        if not self.root:
            return []
        return self.root.get_all_paths_to_leaves()

    def get_random_path(self, language: str = "english") -> TagTrace:
        """Get a random path from root to a leaf node (uniform sampling)"""
        all_paths = self.get_all_paths()
        if not all_paths:
            raise ValueError(f"No paths available in tree for {self.framework_name}")

        random_path = random.choice(all_paths)
        return TagTrace(nodes=random_path, language=language)

    def get_weighted_random_path(
        self, sampling_stats: "PathSamplingStats", language: str = "english"
    ) -> TagTrace:
        """
        Get a weighted random path prioritizing less-sampled paths.

        Args:
            sampling_stats: PathSamplingStats instance for tracking
            language: Language for labels

        Returns:
            TagTrace object with selected path
        """
        all_paths = self.get_all_paths()
        if not all_paths:
            raise ValueError(f"No paths available in tree for {self.framework_name}")

        # Generate path IDs
        path_ids = [self._get_path_id(path) for path in all_paths]

        # Get weighted sample index
        selected_index = sampling_stats.sample_path_index(path_ids)
        selected_path = all_paths[selected_index]

        # Record the sample
        sampling_stats.record_sample(path_ids[selected_index])

        return TagTrace(nodes=selected_path, language=language)

    def _get_path_id(self, path: List[ProblemTypeNode]) -> str:
        """Generate unique ID for a path"""
        return "→".join(node.id for node in path)

    def add_new_child_node(
        self, parent_id: str, new_tag_data: Dict[str, str]
    ) -> Optional[ProblemTypeNode]:
        """
        Add a new child node to the tree.

        Args:
            parent_id: ID of the parent node
            new_tag_data: Dict with 'id', 'en', 'zh' keys

        Returns:
            The new ProblemTypeNode if successful, None otherwise
        """
        parent_node = self.get_node_by_id(parent_id)
        if not parent_node:
            logger.error(f"Parent node not found: {parent_id}")
            return None

        # Create new node
        new_node = ProblemTypeNode(
            id=new_tag_data["id"],
            en=new_tag_data["en"],
            zh=new_tag_data["zh"],
            parent=parent_node,
        )

        # Add to parent's children
        parent_node.children.append(new_node)

        # Add to index
        self.node_index[new_node.id] = new_node

        logger.info(f"Added new node '{new_node.en}' as child of '{parent_node.en}'")

        return new_node

    def save_to_file(self, filepath: str):
        """Save tree structure to JSON file"""
        if not self.root:
            logger.error("Cannot save tree: no root node")
            return

        data = {"framework": self.framework_name, "tree": self.root.to_dict()}

        # Create backup of existing file
        path = Path(filepath)
        if path.exists():
            backup_path = path.with_suffix(".json.bak")
            path.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved problem type tree to {filepath}")

    def get_node_by_id(self, node_id: str) -> Optional[ProblemTypeNode]:
        """Get node by its ID"""
        return self.node_index.get(node_id)

    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get statistics about the tree"""
        if not self.root:
            return {"total_nodes": 0, "total_paths": 0, "max_depth": 0}

        all_paths = self.get_all_paths()
        depths = [len(path) for path in all_paths]

        return {
            "total_nodes": len(self.node_index),
            "total_paths": len(all_paths),
            "max_depth": max(depths) if depths else 0,
            "min_depth": min(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
        }


class ProblemTypeTreeManager:
    """
    Manages problem type trees for multiple frameworks.
    """

    def __init__(self, frameworks_dir: str, llm_client=None, framework_manager=None):
        self.frameworks_dir = Path(frameworks_dir)
        self.trees: Dict[str, ProblemTypeTree] = {}

        # Sampling statistics for each framework
        self.sampling_stats: Dict[str, PathSamplingStats] = {}

        # Framework manager for accessing framework configurations
        self.framework_manager = framework_manager

        # New tag generator (requires LLM client)
        self.llm_client = llm_client
        self.tag_generator = NewTagGenerator(llm_client) if llm_client else None

    def load_framework_tree(self, framework_name: str) -> ProblemTypeTree:
        """Load problem type tree for a specific framework"""
        if framework_name in self.trees:
            return self.trees[framework_name]

        # Look for problem_types.json in framework directory
        tree_file = self.frameworks_dir / framework_name / "problem_types.json"

        if not tree_file.exists():
            raise FileNotFoundError(f"Problem type tree file not found: {tree_file}")

        tree = ProblemTypeTree(framework_name)
        tree.load_from_file(str(tree_file))

        self.trees[framework_name] = tree

        # Initialize sampling stats for this framework
        if framework_name not in self.sampling_stats:
            self.sampling_stats[framework_name] = PathSamplingStats(
                framework_name=framework_name, stats_dir=str(self.frameworks_dir)
            )

        return tree

    def get_tree(self, framework_name: str) -> Optional[ProblemTypeTree]:
        """Get loaded tree for framework"""
        return self.trees.get(framework_name)

    def generate_tag_trace_for_persona(
        self, framework_name: str, persona: str, language: str = "english"
    ) -> TagTrace:
        """
        Generate a tag trace from the problem type tree with intelligent sampling.

        This method now uses:
        1. Weighted sampling that prioritizes less-sampled paths
        2. Small probability to generate and add new problem type tags

        Args:
            framework_name: Name of the framework
            persona: Persona description (for future persona-aware selection)
            language: Language for labels ("english" or "chinese")

        Returns:
            TagTrace object with selected path
        """
        # Always reload tree from file to ensure we see tags added by other processes
        tree = self.load_framework_tree(framework_name)

        # Get sampling stats for this framework
        sampling_stats = self.sampling_stats.get(framework_name)
        if not sampling_stats:
            # Initialize if not already loaded
            sampling_stats = PathSamplingStats(
                framework_name=framework_name, stats_dir=str(self.frameworks_dir)
            )
            self.sampling_stats[framework_name] = sampling_stats

        # Check if we should generate a new tag (before sampling)
        if self.tag_generator and self.tag_generator.should_generate_new_tag():
            new_node = self._attempt_new_tag_generation(tree, language)
            # If a new tag was generated, update the cached tree reference
            if new_node:
                self.trees[framework_name] = tree
                logger.debug(
                    f"Updated cached tree for {framework_name} after new tag generation"
                )

        # Use weighted sampling to prioritize less-sampled paths
        trace = tree.get_weighted_random_path(
            sampling_stats=sampling_stats, language=language
        )

        # Save sampling stats periodically (every 10 samples)
        if sampling_stats.total_samples % 10 == 0:
            sampling_stats.save_stats()

        logger.info(
            f"Generated trace for {framework_name}: " f"{trace.get_label_string()}"
        )

        return trace

    def _attempt_new_tag_generation(
        self, tree: ProblemTypeTree, language: str
    ) -> Optional[ProblemTypeNode]:
        """
        Attempt to generate and add a new problem type tag to the tree.

        Uses file-based locking to ensure thread/process safety when modifying
        the tag tree in multi-process environments.

        Args:
            tree: The ProblemTypeTree to expand
            language: Language for generation

        Returns:
            The new node if successful, None otherwise
        """
        try:
            # Acquire lock for tag tree modification
            with tag_tree_lock(
                tree.framework_name, str(self.frameworks_dir), timeout=30.0
            ):
                # Reload tree from file to get latest version
                # (another process might have modified it while we were waiting for lock)
                tree_file = (
                    self.frameworks_dir / tree.framework_name / "problem_types.json"
                )
                tree.load_from_file(str(tree_file))

                # Select a random intermediate or leaf node as parent
                all_paths = tree.get_all_paths()
                if not all_paths:
                    return None

                # Choose a random path
                random_path = random.choice(all_paths)

                # Select parent: prefer intermediate nodes, but allow leaf nodes too
                if len(random_path) > 1:
                    # Exclude root (index 0), include all others
                    parent_node = random.choice(random_path[1:])
                else:
                    parent_node = random_path[-1]

                # Get existing siblings
                existing_siblings = parent_node.children

                # Get framework config for capability constraints
                framework_config = None
                if self.framework_manager:
                    framework_config = self.framework_manager.get_framework(
                        tree.framework_name
                    )

                # Generate new tag
                new_tag_data = self.tag_generator.generate_new_tag(
                    parent_node=parent_node,
                    existing_siblings=existing_siblings,
                    framework_config=framework_config,
                    language=language,
                )

                if not new_tag_data:
                    logger.warning("Failed to generate new tag data")
                    return None

                # Add new node to tree
                new_node = tree.add_new_child_node(
                    parent_id=parent_node.id, new_tag_data=new_tag_data
                )

                if new_node:
                    # Save the expanded tree (still within lock)
                    tree.save_to_file(str(tree_file))

                    logger.info(
                        f"🌱 NEW TAG GENERATED AND ADDED: {new_node.en} "
                        f"(parent: {parent_node.en}, total nodes: {len(tree.node_index)})"
                    )

                return new_node

        except TimeoutError as e:
            logger.error(f"Failed to acquire tag tree lock: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate new tag: {e}")
            return None

    def select_problem_type_from_trace(
        self, trace: TagTrace, persona: str, language: str = "english"
    ) -> Tuple[ProblemTypeNode, str]:
        """
        Select a specific problem type from the trace by truncating at a random depth.

        The trace will be truncated at a random node, ensuring at least 2 problem type
        levels (root + level1 + level2). The last node of the truncated trace becomes
        the selected problem type.

        Args:
            trace: TagTrace object (will be modified in place by truncation)
            persona: Persona description (for future persona-aware selection)
            language: Language for label

        Returns:
            Tuple of (selected_node, label_string)
        """
        import random

        if not trace.nodes or len(trace.nodes) < 3:
            raise ValueError("Trace must have at least 3 nodes (root + 2 levels)")

        # Randomly select truncation depth
        # Index 2 means keep root + 2 levels (minimum), Index len-1 means keep all (leaf)
        truncation_index = random.randint(2, len(trace.nodes) - 1)

        # Truncate trace to selected depth (keep nodes from 0 to truncation_index inclusive)
        trace.nodes = trace.nodes[: truncation_index + 1]

        # The last node after truncation is the selected problem type
        selected_node = trace.nodes[-1]
        label = selected_node.get_label(language)

        logger.info(
            f"Truncated trace to depth {truncation_index + 1}, "
            f"selected problem type: {label} for persona: {persona[:50]}..."
        )

        return selected_node, label

    def get_all_framework_statistics(self) -> Dict[str, Any]:
        """Get statistics for all loaded frameworks"""
        stats = {}
        for framework_name, tree in self.trees.items():
            stats[framework_name] = tree.get_tree_statistics()
        return stats
