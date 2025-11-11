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
Lightweight .env loader to populate environment variables from a file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[Path]) -> None:
    """
    Load key=value pairs from a .env file into os.environ.

    Existing environment variables take precedence and will not be overwritten.
    Lines beginning with '#' or blank lines are ignored.
    """
    if not env_path:
        return
    env_path = Path(env_path)
    if not env_path.exists():
        return

    try:
        with env_path.open("r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # Silently ignore IO errors; upstream code can handle missing vars.
        return
