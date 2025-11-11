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
import datetime
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

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
langfuse_client = get_langfuse_client()


def parse_datetime(value: str) -> datetime.datetime:
    trimmed = value.strip()
    try:
        return datetime.datetime.fromisoformat(trimmed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Expected ISO format like 2025-10-29 09:48:25.681"
        ) from exc


def sanitize_trace_name(raw_name: Optional[str]) -> str:
    if not raw_name:
        return "unnamed_trace"
    sanitized = re.sub(r"[^\w.-]+", "_", raw_name.strip())
    sanitized = sanitized.strip("._")
    if not sanitized:
        return "unnamed_trace"
    return sanitized[:100]


def get_trace_jsonl(trace_id: str) -> Optional[str]:
    """Fetches a trace by ID and returns the JSONL content as a string."""
    return _get_trace_jsonl(langfuse_client, trace_id)


def get_trace_and_save(trace_id: str, output_filename: str) -> bool:
    """Fetches a trace by ID and saves it to the specified file. Returns True if successful."""
    print(f"🔍 Attempting to fetch trace {trace_id} and save to {output_filename}...")
    return _get_trace_and_save(
        langfuse_client, trace_id, output_filename, LANGFUSE_HOST
    )


def get_langfuse_trace_by_id(trace_id: str) -> Optional[str]:
    """
    获取指定trace_id的Langfuse trace并保存到文件

    Args:
        trace_id: Langfuse trace ID

    Returns:
        str: 保存的文件路径，失败时返回None
    """
    return _get_langfuse_trace_by_id(langfuse_client, trace_id)


def get_langfuse_traces_by_time(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    output_dir: str,
    concurrency: int = 1,
) -> None:
    """
    获取指定时间范围内的Langfuse traces并保存到文件夹中
    """
    if concurrency < 1:
        raise ValueError("concurrency 必须大于等于 1。")

    output_path = Path(output_dir).expanduser()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"无法创建输出文件夹 {output_path}: {exc}") from exc

    page = 1
    executor: Optional[ThreadPoolExecutor] = None
    pending_tasks: List[Dict[str, object]] = []

    count: Dict[str, int] = {
        "skipped": 0,
        "exists": 0,
        "saved": 0,
        "failed": 0,
    }
    try:
        while True:
            traces = langfuse_client.api.trace.list(
                from_timestamp=start_time,
                to_timestamp=end_time,
                page=page,
            )
            trace_dict = traces.dict() if traces and hasattr(traces, "dict") else {}
            traces_data = trace_dict.get("data") or []
            if not traces_data:
                break

            print(f"📄 Processing {len(traces_data)} trace(s) on page {page}...")
            page += 1
            for trace in traces_data:
                trace_id = trace.get("id")
                trace_name = trace.get("name")
                if not trace_id or not trace_name or not trace.get("output"):
                    print(
                        f"⚠️ Skipping trace without ID or name or output: {trace_id}",
                        file=sys.stderr,
                    )
                    count["skipped"] += 1
                    continue

                safe_name = sanitize_trace_name(trace_name)
                trace_folder = output_path / safe_name
                try:
                    trace_folder.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    print(
                        f"❌ 无法创建trace目录 {trace_folder}: {exc}",
                        file=sys.stderr,
                    )
                    count["failed"] += 1
                    continue

                output_file = trace_folder / f"{trace_id}.jsonl"
                if os.path.exists(output_file):
                    print(f"💾 Trace {trace_id} already exists, skipping...")
                    count["exists"] += 1
                    continue

                if concurrency > 1:
                    if executor is None:
                        executor = ThreadPoolExecutor(max_workers=concurrency)
                    future = executor.submit(
                        get_trace_and_save, trace_id, str(output_file)
                    )
                    pending_tasks.append(
                        {
                            "future": future,
                            "trace_id": trace_id,
                            "trace_name": trace_name,
                            "output_file": str(output_file),
                        }
                    )
                else:
                    saved = get_trace_and_save(trace_id, str(output_file))
                    if saved:
                        count["saved"] += 1
                    else:
                        count["failed"] += 1
    finally:
        if executor:
            executor.shutdown(wait=True)

    if executor:
        for task in pending_tasks:
            trace_id = task["trace_id"]
            try:
                saved = bool(task["future"].result())
            except Exception as exc:
                print(f"❌ 下载 trace {trace_id} 失败: {exc}", file=sys.stderr)
                count["failed"] += 1
                continue

            if saved:
                count["saved"] += 1
            else:
                print(
                    f"❌ 下载 trace {trace_id} 失败: get_trace_and_save 返回 False",
                    file=sys.stderr,
                )
                count["failed"] += 1

    print(
        f"💾 成功下载 {count['saved']} 个trace，失败 {count['failed']} 个，跳过 {count['skipped']} 个，已存在 {count['exists']} 个。"
    )


def main():
    parser = argparse.ArgumentParser(
        description=("按时间范围批量获取Langfuse traces，并保存为 JSONL 文件。")
    )
    parser.add_argument(
        "--start",
        required=True,
        type=parse_datetime,
        help="起始时间，例如 '2025-10-29 09:48:25.681'",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=parse_datetime,
        help="结束时间，例如 '2025-10-29 10:48:25.681'",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出文件夹路径，trace文件将保存在此目录下的子目录中。",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发下载的线程数量，默认 1 表示串行执行。",
    )

    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("结束时间必须晚于起始时间。")
    if args.concurrency < 1:
        parser.error("并发数量必须大于等于 1。")

    try:
        get_langfuse_traces_by_time(
            args.start,
            args.end,
            args.output_dir,
            concurrency=args.concurrency,
        )
    except Exception as exc:
        print(f"❌ 执行失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
