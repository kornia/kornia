"""CLI interface for Kornia MCP server."""

from typing import List
from types import ModuleType
import logging
import inspect

import kornia
from kornia.mcp.server import add_func_as_tool, add_class_as_tool
from kornia.core.external import mcp as _mcp

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
    "Normalize"
]


VALID_FEATURE_FUNCTIONS = [
    "HardNet8",
    "LightGlue",
]


def add_tools_from_module(
    mcp: "_mcp.server.fastmcp.FastMCP",
    module: ModuleType,
    tool_prefix: str,
    valid_functions: List[str]
) -> "_mcp.server.fastmcp.FastMCP":
    """Add tools from a module to the MCP server."""
    # Register functions from enhance module
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if not name.startswith('_'):  # Skip private functions
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


if __name__ == '__main__':
    mcp = main()
    mcp.run(transport="sse")
