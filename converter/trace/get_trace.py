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

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Import common utilities (support both relative and absolute import)
try:
    from .langfuse_utils import get_langfuse_client
    from .langfuse_utils import get_langfuse_trace_by_id as _get_langfuse_trace_by_id
    from .langfuse_utils import get_trace_and_save as _get_trace_and_save
    from .langfuse_utils import get_trace_jsonl as _get_trace_jsonl
    from .langfuse_utils import load_langfuse_config
except ImportError:
    from langfuse_utils import get_langfuse_client
    from langfuse_utils import get_langfuse_trace_by_id as _get_langfuse_trace_by_id
    from langfuse_utils import get_trace_and_save as _get_trace_and_save
    from langfuse_utils import get_trace_jsonl as _get_trace_jsonl
    from langfuse_utils import load_langfuse_config


# 加载配置
LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST = load_langfuse_config()

# 创建全局 Langfuse 客户端
langfuse = get_langfuse_client()


def draw_spans(spans_data):
    """Draw a visual hierarchy tree of spans"""
    for i, span in enumerate(spans_data):

        span_type = span.get("type", "unknown").upper()
        span_name = span.get("name", "Unnamed")
        span_id = span.get("id", "Unnamed")

        # Format with color codes (works in most terminals)
        type_color = {"GENERATION": "🤖", "SPAN": "📊", "EVENT": "⚡", "UNKNOWN": "❓"}

        icon = type_color.get(span_type, "📋")
        print(f"{str(i+1).zfill(4)} {icon} [{span_type}] {span_name} <{span_id}>")


# Save all spans to a separate JSONL file
def save_spans_to_jsonl(trace_id: str, filename="spans_output.jsonl") -> str:
    """Save all individual spans to a JSONL file with detailed information"""
    print(f"\n📊 Saving spans to JSONL file: {filename}...")

    span_count = 0
    with open(filename, "w", encoding="utf-8") as f:
        # Try to fetch detailed observations for this trace
        try:
            page = 1
            spans_data = []
            while True:
                print(f"Fetching observations for trace {trace_id}, page {page}...")
                observations = langfuse.api.observations.get_many(
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
                draw_spans(spans_data)

                # Save each span as a separate record
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
                            span.get("endTime").isoformat()
                            if span.get("endTime")
                            else None
                        ),
                        "usage": span.get("usage"),
                        "metadata": span.get("metadata"),
                        "parentObservationId": span.get("parentObservationId"),
                        "level": span.get("level", 0),
                    }

                    # Write as a single line of JSON
                    f.write(json.dumps(span_record, ensure_ascii=False) + "\n")
                    span_count += 1

        except Exception as e:
            print(
                f"⚠️  Could not fetch observations for trace {trace_id}: {str(e)[:50]}..."
            )
            return None


def get_trace_jsonl(trace_id: str) -> Optional[str]:
    """Fetches a trace by ID and returns the JSONL content as a string."""
    return _get_trace_jsonl(langfuse, trace_id)


def get_trace_and_save(trace_id: str, output_filename: str) -> bool:
    """Fetches a trace by ID and saves it to the specified file. Returns True if successful."""
    return _get_trace_and_save(langfuse, trace_id, output_filename, LANGFUSE_HOST)


def get_langfuse_trace_by_id(trace_id: str) -> Optional[str]:
    """
    获取指定trace_id的Langfuse trace并保存到文件

    Args:
        trace_id: Langfuse trace ID

    Returns:
        str: 保存的文件路径，失败时返回None
    """
    return _get_langfuse_trace_by_id(langfuse, trace_id)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Langfuse traces by ID and export them as JSONL format."
    )
    parser.add_argument("trace_id", help="The trace ID to fetch")
    parser.add_argument(
        "-o", "--output", help="Output filename (default: <trace_id>.jsonl)"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print to stdout only, do not save to file",
    )

    args = parser.parse_args()

    if args.print_only:
        # Print to stdout only
        jsonl_content = get_trace_jsonl(args.trace_id)
        if jsonl_content:
            print(jsonl_content)
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Save to file
        output_filename = args.output if args.output else f"{args.trace_id}.jsonl"
        success = get_trace_and_save(args.trace_id, output_filename)
        if success:
            print(f"Successfully saved trace to {output_filename}")
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
