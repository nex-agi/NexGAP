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
URL Validator for Gyrfalcon v5

Validates URLs by checking accessibility via HTTP requests.
"""

import logging
import os
from typing import Any, Dict, List

import requests

from .base import Agent, AgentContext, AgentOutput

logger = logging.getLogger(__name__)


class URLValidatorAgent(Agent):
    """
    Validates URLs by sending HTTP HEAD requests to check accessibility.
    """

    def __init__(
        self, name: str = "URLValidatorAgent", max_retries: int = 2, timeout: int = 10
    ):
        super().__init__(name)
        self.max_retries = max_retries
        self.timeout = timeout
        self.proxies = {
            "http": os.getenv("HTTP_PROXY"),
            "https": os.getenv("HTTPS_PROXY"),
        }

    def run(self, context: AgentContext) -> AgentOutput:
        """
        Validate extracted URLs.

        Expected context keys:
            - extracted_urls: List[Dict] (URLs to validate)

        Adds to context:
            - validation_results: List[Dict] (validation results for each URL)
            - accessible_count: int (number of accessible URLs)
            - broken_urls: List[Dict] (URLs that are not accessible)
        """
        extracted_urls = context.get("extracted_urls", [])

        if not extracted_urls:
            logger.info("No URLs to validate")
            return AgentOutput(
                success=True,
                data={
                    "validation_results": [],
                    "accessible_count": 0,
                    "broken_urls": [],
                },
            )

        try:
            logger.info(f"Validating {len(extracted_urls)} URL(s)...")

            validation_results = []
            accessible_count = 0
            broken_urls = []

            for url_info in extracted_urls:
                url = url_info.get("url")
                if not url:
                    continue

                validation = self._validate_url(url)
                validation_results.append(validation)

                if validation["accessible"]:
                    accessible_count += 1
                else:
                    broken_urls.append(
                        {
                            **url_info,
                            "error": validation["error"],
                            "status_code": validation.get("status_code"),
                        }
                    )

            logger.info(
                f"Validation complete: {accessible_count}/{len(extracted_urls)} accessible"
            )

            return AgentOutput(
                success=True,
                data={
                    "validation_results": validation_results,
                    "accessible_count": accessible_count,
                    "broken_urls": broken_urls,
                },
            )

        except Exception as e:
            logger.error(f"URL validation failed: {e}")
            return AgentOutput(
                success=False, errors=[f"URL validation error: {str(e)}"]
            )

    def _validate_url(self, url: str) -> Dict[str, Any]:
        """Validate a single URL with retry logic"""

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Validating URL (attempt {attempt}/{self.max_retries}): {url}"
                )

                response = requests.head(
                    url,
                    allow_redirects=True,
                    timeout=self.timeout,
                    proxies=self.proxies,
                    headers={"User-Agent": "Mozilla/5.0"},
                )

                if response.status_code == 200:
                    logger.info(f"  ✓ Accessible (status: {response.status_code})")
                    return {
                        "url": url,
                        "accessible": True,
                        "status_code": response.status_code,
                        "error": None,
                        "redirect_url": response.url if response.url != url else None,
                        "content_type": response.headers.get("content-type"),
                    }
                else:
                    logger.warning(
                        f"  ✗ Not accessible (status: {response.status_code})"
                    )
                    if attempt < self.max_retries:
                        continue

                    return {
                        "url": url,
                        "accessible": False,
                        "status_code": response.status_code,
                        "error": f"HTTP {response.status_code}",
                        "redirect_url": None,
                        "content_type": None,
                    }

            except requests.exceptions.Timeout:
                logger.warning(f"  ✗ Timeout after {self.timeout}s")
                if attempt < self.max_retries:
                    continue

                return {
                    "url": url,
                    "accessible": False,
                    "status_code": None,
                    "error": "Request timeout",
                    "redirect_url": None,
                    "content_type": None,
                }

            except requests.exceptions.RequestException as e:
                logger.warning(f"  ✗ Request failed: {str(e)}")
                if attempt < self.max_retries:
                    continue

                return {
                    "url": url,
                    "accessible": False,
                    "status_code": None,
                    "error": str(e),
                    "redirect_url": None,
                    "content_type": None,
                }

        # Should never reach here
        return {
            "url": url,
            "accessible": False,
            "status_code": None,
            "error": "Max retries exceeded",
            "redirect_url": None,
            "content_type": None,
        }
