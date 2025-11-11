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
Schema Validation Example

This script demonstrates how to use the Pydantic schemas to validate
framework_config and nexau_config files.
"""

import json
from pathlib import Path
from typing import Union

import yaml
from schema import FrameworkConfig, NexauConfig


def validate_framework_config(config_path: Union[str, Path]) -> bool:
    """验证 framework_config 文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # 使用 Pydantic 模型验证
        framework_config = FrameworkConfig(**config_data)

        print(f"✅ Framework config validation passed: {config_path}")
        print(f"   Agents count: {len(framework_config.agents)}")
        print(
            f"   Tools count: {len(framework_config.tools) if framework_config.tools else 0}"
        )
        print(
            f"   MCPs count: {len(framework_config.mcps) if framework_config.mcps else 0}"
        )

        return True

    except Exception as e:
        print(f"❌ Framework config validation failed: {config_path}")
        print(f"   Error: {e}")
        return False


def validate_nexau_config(config_path: Union[str, Path]) -> bool:
    """验证 nexau_config 文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # 使用 Pydantic 模型验证
        nexau_config = NexauConfig(**config_data)

        print(f"✅ Nexau config validation passed: {config_path}")
        print(f"   Agent name: {nexau_config.name}")
        print(f"   Max context: {nexau_config.max_context}")
        print(f"   Tools count: {len(nexau_config.tools)}")
        print(f"   Sub-agents count: {len(nexau_config.sub_agents)}")

        return True

    except Exception as e:
        print(f"❌ Nexau config validation failed: {config_path}")
        print(f"   Error: {e}")
        return False


def export_json_schema(output_dir: Union[str, Path] = "schema_exports"):
    """导出 JSON Schema 文件"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 导出 Framework Config Schema
    framework_schema = FrameworkConfig.schema()
    with open(output_path / "framework_config.schema.json", "w", encoding="utf-8") as f:
        json.dump(framework_schema, f, indent=2, ensure_ascii=False)

    # 导出 Nexau Config Schema
    nexau_schema = NexauConfig.schema()
    with open(output_path / "nexau_config.schema.json", "w", encoding="utf-8") as f:
        json.dump(nexau_schema, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON schemas exported to {output_path}")


def main():
    """主函数"""
    print("=== Config Schema Validation Example ===\n")

    # 验证测试文件
    test_files = [
        ("../test_framework_config.yaml", validate_framework_config),
        ("../test_framework_config_with_capabilities.yaml", validate_framework_config),
        ("../test_nexau_config.yaml", validate_nexau_config),
        ("../output_nexau_config_with_capabilities_fixed.yaml", validate_nexau_config),
    ]

    success_count = 0
    total_count = len(test_files)

    for file_path, validator_func in test_files:
        if Path(file_path).exists():
            if validator_func(file_path):
                success_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
        print()

    print(f"Validation Results: {success_count}/{total_count} passed")

    # 导出 JSON Schema
    print("\n=== Exporting JSON Schemas ===")
    export_json_schema()


if __name__ == "__main__":
    main()
