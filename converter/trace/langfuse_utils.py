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
Langfuse common utilities for trace fetching and configuration
Shared by get_trace.py and get_traces.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langfuse import Langfuse


def load_langfuse_config():
    """Read Langfuse configuration from .env file in NexGAP root directory"""
    # Get project root directory (NexGAP directory)
    current_file = Path(__file__).resolve()
    project_root = (
        current_file.parent.parent.parent
    )  # converter/trace/langfuse_utils.py -> NexGAP/
    env_path = project_root / ".env"

    if env_path.exists():
        # Load .env file
        load_dotenv(env_path)
        print(f"✅ Loaded Langfuse configuration from {env_path}")
    else:
        print(f"⚠️ .env file not found: {env_path}")
        print("Using default configuration or environment variables")

    # Read configuration from environment variables
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST")

    if not secret_key or not public_key or not host:
        raise ValueError(
            "Missing required Langfuse environment variables: LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST"
        )

    return secret_key, public_key, host


def get_langfuse_client() -> Langfuse:
    """Create and return Langfuse client instance"""
    secret_key, public_key, host = load_langfuse_config()
    return Langfuse(
        secret_key=secret_key,
        public_key=public_key,
        host=host,
        timeout=30,
    )


def get_trace_jsonl(langfuse_client: Langfuse, trace_id: str) -> Optional[str]:
    """Fetches a trace by ID and returns the JSONL content as a string.

    Args:
        langfuse_client: Langfuse client instance
        trace_id: Trace ID to fetch

    Returns:
        JSONL content as string, or None if failed
    """
    jsonl_lines = []

    try:
        page = 1
        spans_data = []
        while True:
            print(f"Fetching observations for trace {trace_id}, page {page}...")
            observations = langfuse_client.api.observations.get_many(
                trace_id=trace_id, page=page
            )
            if (
                observations.dict()
                and "data" in observations.dict()
                and len(observations.dict()["data"]) > 0
            ):
                spans_data.extend(observations.dict()["data"])
                page += 1
            else:
                break

        if len(spans_data) > 0:
            spans_data.sort(key=lambda x: x.get("startTime"))

            for span in spans_data:
                span_record = {
                    "trace_id": trace_id,
                    "span_id": span.get("id"),
                    "span_type": span.get("type"),
                    "span_name": span.get("name"),
                    "model": span.get("model"),
                    "input": span.get("input"),
                    "output": span.get("output"),
                    "startTime": (
                        span.get("startTime").isoformat()
                        if span.get("startTime")
                        else None
                    ),
                    "endTime": (
                        span.get("endTime").isoformat() if span.get("endTime") else None
                    ),
                    "usage": span.get("usage"),
                    "metadata": span.get("metadata"),
                    "parentObservationId": span.get("parentObservationId"),
                    "level": span.get("level", 0),
                }
                jsonl_lines.append(json.dumps(span_record, ensure_ascii=False))

        return "\n".join(jsonl_lines) if jsonl_lines else None

    except Exception as e:
        print(f"Error fetching trace {trace_id}: {str(e)}", file=sys.stderr)
        return None


def get_trace_and_save(
    langfuse_client: Langfuse,
    trace_id: str,
    output_filename: str,
    langfuse_host: str = None,
) -> bool:
    """Fetches a trace by ID and saves it to the specified file.

    Args:
        langfuse_client: Langfuse client instance
        trace_id: Trace ID to fetch
        output_filename: Output file path
        langfuse_host: Langfuse host URL (for display only)

    Returns:
        True if successful, False otherwise
    """
    print(f"🔍 Attempting to fetch trace: {trace_id}")
    if langfuse_host:
        print(f"📍 Using Langfuse host: {langfuse_host}")

    # Add retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        print(f"📡 Attempt {attempt + 1}/{max_retries} to fetch trace...")

        jsonl_content = get_trace_jsonl(langfuse_client, trace_id)
        if jsonl_content is not None:
            try:
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(jsonl_content + "\n")
                print(f"✅ Successfully saved trace to {output_filename}")
                return True
            except Exception as e:
                print(
                    f"❌ Error saving to file {output_filename}: {str(e)}",
                    file=sys.stderr,
                )
                return False
        else:
            if attempt < max_retries - 1:
                print(f"⏳ Trace not found, waiting 5 seconds before retry...")
                import time

                time.sleep(5)
            else:
                print(f"❌ Failed to fetch trace after {max_retries} attempts")

    return False


def get_langfuse_trace_by_id(langfuse_client: Langfuse, trace_id: str) -> Optional[str]:
    """
    Get Langfuse trace by trace_id and save to temporary file

    Args:
        langfuse_client: Langfuse client instance
        trace_id: Langfuse trace ID

    Returns:
        str: Path to saved file, None on failure
    """
    import os
    import tempfile
    from datetime import datetime

    # Generate temporary filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = f"langfuse_trace_{trace_id[:8]}_{timestamp}.jsonl"

    # Use temporary directory
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, temp_filename)

    # Get and save trace
    success = get_trace_and_save(langfuse_client, trace_id, output_path)

    if success:
        print(f"✅ Langfuse trace saved to: {output_path}")
        return output_path
    else:
        print(f"❌ Failed to save Langfuse trace for ID: {trace_id}")
        return None
