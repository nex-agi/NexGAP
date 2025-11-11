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
Agent responsible for planning and downloading supporting files for queries.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import unquote_to_bytes, urlparse

try:
    import requests
except (
    ImportError
):  # pragma: no cover - requests should be available but guard for safety
    requests = None

from .base import Agent, AgentContext, AgentOutput

if TYPE_CHECKING:
    from llm_interface.query_generator import LLMClient, PromptTemplateManager

logger = logging.getLogger(__name__)
FILE_TAG = "📁 >>>FILES<<<"


class FileSystemAgent(Agent):
    """Generates file download plans and retrieves assets for file-dependent queries."""

    def __init__(
        self,
        llm_client: "LLMClient",
        prompt_manager: "PromptTemplateManager",
        base_dir: Path,
    ):
        super().__init__("file_system")
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = self.base_dir.parent

    def run(self, context: AgentContext) -> AgentOutput:
        if not context.get("query_requires_files"):
            logger.debug("%s Skipped (query does not require files)", FILE_TAG)
            return AgentOutput(success=True)

        query_text: Optional[str] = context.get("current_query_text")
        required_items: List[str] = context.get("query_required_items", [])
        framework_name: Optional[str] = context.get("framework_name")
        language: str = (context.get("language") or "english").lower()

        if not query_text or not framework_name:
            error = "FileSystemAgent needs query text and framework name."
            logger.error("%s %s", FILE_TAG, error)
            return AgentOutput(success=False, errors=[error])

        prompt = self.prompt_manager.format_file_system_plan_prompt(
            query=query_text, required_items=required_items, language=language
        )

        plan_start = time.time()
        try:
            response = self.llm_client.generate_completion(prompt)
            plan_elapsed = time.time() - plan_start

            plan_info = self._parse_plan(response)
            files_to_fetch = (
                plan_info.get("files", []) if isinstance(plan_info, dict) else []
            )
            if not files_to_fetch:
                logger.warning("%s LLM plan missing files", FILE_TAG)
                return AgentOutput(
                    success=True,
                    data={"downloaded_files": []},
                    timings={"planning": plan_elapsed},
                )

            download_dir = self._prepare_download_dir(
                framework_name, plan_info.get("directory_name")
            )
            logger.info("%s Using download directory: %s", FILE_TAG, download_dir)
            downloaded_files = self._download_files(files_to_fetch, download_dir)

            context.set("downloaded_files", downloaded_files)
            if plan_info.get("directory_name"):
                context.set("file_plan_directory", plan_info.get("directory_name"))

            return AgentOutput(
                success=True,
                data={
                    "downloaded_files": downloaded_files,
                    "file_download_directory": str(download_dir),
                },
                timings={"planning": plan_elapsed},
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("%s Agent failed: %s", FILE_TAG, exc)
            context.append_error(f"file_system: {exc}")
            return AgentOutput(success=False, errors=[str(exc)])

    def _prepare_download_dir(
        self, framework_name: str, directory_name: Optional[str]
    ) -> Path:
        timestamp = int(time.time() * 1000)
        base = directory_name or f"bundle"
        base = self._sanitize_directory_name(base) or "bundle"
        download_dir = self.base_dir / framework_name / f"{base}-{timestamp}"
        download_dir.mkdir(parents=True, exist_ok=True)
        return download_dir

    def _parse_plan(self, response: str) -> Dict[str, Any]:
        json_str = self._extract_json(response)
        if not json_str:
            return {"files": []}
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning(
                "%s Failed to parse file plan JSON: %s; response=%s",
                FILE_TAG,
                exc,
                response,
            )
            return {"files": []}

        files = data.get("files")
        if not isinstance(files, list):
            files = []

        parsed_files = []
        for entry in files:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url")
            description = entry.get("description", "").strip()
            if not url:
                continue
            parsed_files.append(
                {
                    "url": url,
                    "description": description,
                }
            )

        directory_name = data.get("directory_name")
        directory_name = self._sanitize_directory_name(directory_name)

        return {
            "directory_name": directory_name,
            "files": parsed_files,
        }

    def _download_files(
        self, plan: List[Dict[str, str]], download_dir: Path
    ) -> List[Dict[str, str]]:
        downloaded: List[Dict[str, str]] = []
        for item in plan:
            url = item["url"]
            description = item.get("description", "")
            filename = self._infer_filename(url)
            target_path = download_dir / filename
            scheme = urlparse(url).scheme

            try:
                relative_path = str(target_path.relative_to(self.project_root))
            except ValueError:
                relative_path = str(target_path)

            record = {
                "url": url,
                "description": description,
                "local_path": relative_path,
                "status": "pending",
                "error": None,
                "bundle_directory": (
                    str(download_dir.relative_to(self.project_root))
                    if download_dir
                    else ""
                ),
            }

            try:
                if scheme == "data":
                    logger.info(
                        "%s Decoding data URL into local file: %s",
                        FILE_TAG,
                        target_path,
                    )
                    self._write_data_url(url, target_path)
                else:
                    if scheme != "https":
                        raise ValueError(
                            f"unsupported URL scheme: {scheme or 'unknown'}"
                        )
                    if requests is None:
                        raise RuntimeError("requests library not available")
                    logger.info("%s Downloading supporting file: %s", FILE_TAG, url)
                    response = requests.get(url, timeout=20)
                    response.raise_for_status()
                    with open(target_path, "wb") as f:
                        f.write(response.content)
                record["status"] = "downloaded"
            except Exception as exc:  # noqa: BLE001
                record["status"] = "failed"
                record["error"] = str(exc)
                logger.error("%s Failed to download file %s: %s", FILE_TAG, url, exc)

            downloaded.append(record)

        return downloaded

    @staticmethod
    def _infer_filename(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme == "data":
            media_type = parsed.path.split(";", 1)[0] if parsed.path else ""
            extension_map = {
                "text/plain": ".txt",
                "text/csv": ".csv",
                "text/tab-separated-values": ".tsv",
                "application/json": ".json",
            }
            ext = extension_map.get(media_type, ".txt")
            return f"data_{int(time.time() * 1000)}{ext}"
        fname = os.path.basename(parsed.path)
        if not fname:
            fname = f"file_{int(time.time() * 1000)}"
        return fname

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    @staticmethod
    def _write_data_url(url: str, target_path: Path) -> None:
        header_body = url.split(",", 1)
        if len(header_body) != 2:
            raise ValueError("malformed data URL")
        header, body = header_body
        is_base64 = header.endswith(";base64") or ";base64;" in header
        if is_base64:
            try:
                content = base64.b64decode(body, validate=True)
            except Exception as exc:
                raise ValueError(f"invalid base64 data URL: {exc}") from exc
        else:
            content = unquote_to_bytes(body)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as fh:
            fh.write(content)

    @staticmethod
    def _sanitize_directory_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        name = name.strip().lower()
        name = re.sub(r"[^a-z0-9\-]+", "-", name)
        name = re.sub(r"-+", "-", name).strip("-")
        if not name:
            return None
        return name[:60]
