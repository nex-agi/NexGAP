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
Cache Manager for NexGAP Converter
Manages various caches during the conversion process to avoid redundant loading and processing
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class CacheManager:
    """Cache Manager"""

    def __init__(self, cache_dir: Path = None):
        """
        Initialize cache manager

        Args:
            cache_dir: Cache directory, defaults to converter/.cache
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Different types of cache files
        self.mcp_cache_file = self.cache_dir / "mcp_tools_cache.json"
        self.agent_tools_cache_file = self.cache_dir / "agent_tools_cache.json"
        self.processed_traces_file = self.cache_dir / "processed_traces.json"

    def get_mcp_cache(self) -> Dict[str, list]:
        """Get MCP tools cache"""
        return self._load_cache(self.mcp_cache_file)

    def set_mcp_cache(self, cache_data: Dict[str, list]):
        """Save MCP tools cache"""
        self._save_cache(self.mcp_cache_file, cache_data)

    def get_agent_tools_cache(self) -> Dict[str, list]:
        """Get Agent→Tools mapping cache"""
        return self._load_cache(self.agent_tools_cache_file)

    def set_agent_tools_cache(self, cache_data: Dict[str, list]):
        """Save Agent→Tools mapping cache"""
        self._save_cache(self.agent_tools_cache_file, cache_data)

    def is_trace_processed(self, trace_id: str) -> bool:
        """Check if trace has been processed"""
        processed = self._load_cache(self.processed_traces_file)
        return trace_id in processed

    def mark_trace_processed(self, trace_id: str, metadata: Dict = None):
        """Mark trace as processed"""
        processed = self._load_cache(self.processed_traces_file)
        processed[trace_id] = {
            "processed_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._save_cache(self.processed_traces_file, processed)

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        if not file_path.exists():
            return ""

        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def is_file_changed(self, file_path: Path, cached_hash: str = None) -> bool:
        """Check if file has been changed"""
        if cached_hash is None:
            # Get previous hash from cache
            cache_key = f"file_hash_{file_path.name}"
            cache = self._load_cache(self.cache_dir / "file_hashes.json")
            cached_hash = cache.get(str(file_path))

        if not cached_hash:
            return True

        current_hash = self.get_file_hash(file_path)
        return current_hash != cached_hash

    def update_file_hash(self, file_path: Path):
        """Update file hash cache"""
        current_hash = self.get_file_hash(file_path)
        cache = self._load_cache(self.cache_dir / "file_hashes.json")
        cache[str(file_path)] = current_hash
        self._save_cache(self.cache_dir / "file_hashes.json", cache)

    def clear_all(self):
        """Clear all caches"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        print(f"✅ Cleared all caches: {self.cache_dir}")

    def clear_mcp_cache(self):
        """Clear MCP cache"""
        if self.mcp_cache_file.exists():
            self.mcp_cache_file.unlink()
        if self.agent_tools_cache_file.exists():
            self.agent_tools_cache_file.unlink()
        print("✅ Cleared MCP-related caches")

    def clear_processed_traces(self):
        """Clear processed trace records"""
        if self.processed_traces_file.exists():
            self.processed_traces_file.unlink()
        print("✅ Cleared processed trace records")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "cache_dir": str(self.cache_dir),
            "mcp_tools_count": len(self.get_mcp_cache()),
            "agent_tools_count": len(self.get_agent_tools_cache()),
            "processed_traces_count": len(self._load_cache(self.processed_traces_file)),
            "cache_files": [],
        }

        for cache_file in self.cache_dir.glob("*.json"):
            size_kb = cache_file.stat().st_size / 1024
            stats["cache_files"].append(
                {
                    "name": cache_file.name,
                    "size_kb": round(size_kb, 2),
                    "modified": datetime.fromtimestamp(
                        cache_file.stat().st_mtime
                    ).isoformat(),
                }
            )

        return stats

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load cache file"""
        if not cache_file.exists():
            return {}

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return {}

    def _save_cache(self, cache_file: Path, data: Dict):
        """Save cache file"""
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save cache {cache_file.name}: {e}")


# Global cache manager instance (singleton)
_global_cache_manager = None


def get_cache_manager(cache_dir: Path = None) -> CacheManager:
    """Get global cache manager instance"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(cache_dir)
    return _global_cache_manager
