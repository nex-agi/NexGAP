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
File-based locking mechanism for tag tree synchronization in multi-process scenarios.
"""

import fcntl
import logging
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class FileLock:
    """
    File-based lock for coordinating access to tag tree files across processes.
    Uses fcntl.flock() for POSIX systems.
    """

    def __init__(self, lock_file: str, timeout: float = 30.0):
        """
        Args:
            lock_file: Path to the lock file
            timeout: Maximum time to wait for lock acquisition (seconds)
        """
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self.fd = None

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire the lock.

        Args:
            blocking: If True, wait for lock; if False, return immediately

        Returns:
            True if lock acquired, False otherwise
        """
        # Create lock file directory if needed
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # Open lock file
        self.fd = open(self.lock_file, "w")

        start_time = time.time()

        while True:
            try:
                # Try to acquire exclusive lock
                flags = fcntl.LOCK_EX
                if not blocking:
                    flags |= fcntl.LOCK_NB

                fcntl.flock(self.fd.fileno(), flags)
                logger.debug(f"Acquired lock: {self.lock_file}")
                return True

            except BlockingIOError:
                if not blocking:
                    self.fd.close()
                    self.fd = None
                    return False

                # Check timeout
                if time.time() - start_time > self.timeout:
                    logger.error(f"Lock acquisition timeout: {self.lock_file}")
                    self.fd.close()
                    self.fd = None
                    return False

                # Wait a bit before retrying
                time.sleep(0.1)

    def release(self):
        """Release the lock"""
        if self.fd:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None
            logger.debug(f"Released lock: {self.lock_file}")

    def __enter__(self):
        """Context manager entry"""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
        return False


@contextmanager
def tag_tree_lock(framework_name: str, frameworks_dir: str, timeout: float = 30.0):
    """
    Context manager for acquiring tag tree lock.

    Usage:
        with tag_tree_lock("deer_flow", "/path/to/frameworks"):
            # Safely modify tag tree
            tree.add_new_child_node(...)
            tree.save_to_file(...)

    Args:
        framework_name: Name of the framework
        frameworks_dir: Path to frameworks directory
        timeout: Lock timeout in seconds
    """
    lock_file = Path(frameworks_dir) / framework_name / ".tag_tree.lock"
    lock = FileLock(str(lock_file), timeout=timeout)

    try:
        if lock.acquire(blocking=True):
            yield lock
        else:
            raise TimeoutError(f"Failed to acquire tag tree lock for {framework_name}")
    finally:
        lock.release()


@contextmanager
def sampling_stats_lock(
    framework_name: str, frameworks_dir: str, timeout: float = 10.0
):
    """
    Context manager for acquiring sampling stats lock.

    Usage:
        with sampling_stats_lock("deer_flow", "/path/to/frameworks"):
            # Safely modify sampling stats
            stats.save_stats()

    Args:
        framework_name: Name of the framework
        frameworks_dir: Path to frameworks directory
        timeout: Lock timeout in seconds
    """
    lock_file = Path(frameworks_dir) / framework_name / ".sampling_stats.lock"
    lock = FileLock(str(lock_file), timeout=timeout)

    try:
        if lock.acquire(blocking=True):
            yield lock
        else:
            raise TimeoutError(
                f"Failed to acquire sampling stats lock for {framework_name}"
            )
    finally:
        lock.release()
