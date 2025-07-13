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
from kornia.mcp.server import add_class_as_tool, add_func_as_tool

logger = logging.getLogger(__name__)


VALID_ENHANCE_FUNCTIONS = [
    "AdjustBrightness",
    # "AdjustBrightnessAccumulative",
    "AdjustContrast",
    # "AdjustContrastWithMeanSubtraction",
    "AdjustGamma",
    "AdjustHue",
    "AdjustLog",
    "AdjustSaturation",
    # "AdjustSaturationWithGraySubtraction",
    "AdjustSigmoid",
    "Denormalize",
    "Normalize",
]


VALID_FEATURE_FUNCTIONS = [
    "HardNet8",
    "BlobHessian",
    "CornerGFTT",
    "CornerHarris",
    "BlobDoGSingle",
    "BlobDoG",
    # "LightGlue",
    "HyNet",
    "SOSNet",
    "DiscreteSteerer",
    "DenseSIFTDescriptor",
    "SIFTDescriptor",
    "TFeat",
]

# NOTE: It seems the tuple arguments (e.g. kernel_size) are not
# handeld correctly. The inputSchema may not be correct by this automatic approach.
VALID_FILTER_FUNCTIONS = [
    # "BoxBlur",
    # "BlurPool2D",
    # "DexiNed",
    # "EdgeAwareBlurPool2D",
    # "MaxBlurPool2D",
    # "StableDiffusionDissolving",
    # "GaussianBlur2d",
    # "InRange",
    # "Laplacian",
    # "MedianBlur",
    # "MotionBlur",
    # "Sobel",
    # "UnsharpMask",
]

VALID_GEOMETRY_FUNCTIONS = [
    "RANSAC",
    "Affine",
    "Shear",
    "Rescale",
    "Translate",
    "Rotate",
    "Resize",
    "Hflip",
    "Rot180",
    "Vflip",
]


def add_tools_from_module(
    mcp: "_mcp.server.fastmcp.FastMCP", module: ModuleType, tool_prefix: str, valid_functions: List[str]
) -> "_mcp.server.fastmcp.FastMCP":
    """Add tools from a module to the MCP server."""
    # Register functions from enhance module
    for name in valid_functions:
        if hasattr(module, name):
            obj = getattr(module, name)
            if inspect.isclass(obj):
                print(f"Adding tool: {name}")
                add_class_as_tool(mcp, obj, tool_prefix)
            elif inspect.isfunction(obj):
                print(f"Adding tool: {name}")
                add_func_as_tool(mcp, obj, tool_prefix)


def main():
    """Main entry point for the CLI."""
    mcp = _mcp.server.fastmcp.FastMCP("Kornia MCP Toolbox")
    add_tools_from_module(mcp, kornia.enhance, "kornia_enhance", VALID_ENHANCE_FUNCTIONS)
    add_tools_from_module(mcp, kornia.feature, "kornia_feature", VALID_FEATURE_FUNCTIONS)
    add_tools_from_module(mcp, kornia.filters, "kornia_filters", VALID_FILTER_FUNCTIONS)
    return mcp


if __name__ == "__main__":
    mcp = main()
    mcp.run(transport="sse")
