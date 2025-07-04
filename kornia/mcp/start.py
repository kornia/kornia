# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
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
#

"""CLI interface for Kornia MCP server."""

import inspect
import logging
from types import ModuleType
from typing import List

import kornia
from kornia.core.external import mcp as _mcp
from kornia.mcp.server import add_class_as_tool

logger = logging.getLogger(__name__)


VALID_ENHANCE_FUNCTIONS = [
    "AdjustBrightness",
    "AdjustBrightnessAccumulative",
    "AdjustContrast",
    "AdjustContrastWithMeanSubtraction",
    "AdjustGamma",
    "AdjustHue",
    "AdjustLog",
    "AdjustSaturation",
    "AdjustSaturationWithGraySubtraction",
    "AdjustSigmoid",
    "Denormalize",
    "Normalize",
]


VALID_FEATURE_FUNCTIONS = [
    "HardNet8",
    "LightGlue",
]


def add_tools_from_module(
    mcp: "_mcp.server.fastmcp.FastMCP", module: ModuleType, tool_prefix: str, valid_functions: List[str]
) -> "_mcp.server.fastmcp.FastMCP":
    """Add tools from a module to the MCP server."""
    # Register functions from enhance module
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if not name.startswith("_"):  # Skip private functions
            if name not in valid_functions:
                continue
            print(f"Adding tool: {name}")
            # Create a wrapper function with proper type hints
            add_class_as_tool(mcp, cls, tool_prefix)


def main():
    """Main entry point for the CLI."""
    mcp = _mcp.server.fastmcp.FastMCP("KORNIA")
    add_tools_from_module(mcp, kornia.enhance, "kornia.enhance", VALID_ENHANCE_FUNCTIONS)
    add_tools_from_module(mcp, kornia.feature, "kornia.feature", VALID_FEATURE_FUNCTIONS)

    return mcp


if __name__ == "__main__":
    mcp = main()
    mcp.run(transport="sse")
