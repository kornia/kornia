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

"""MCP Server implementation for Kornia."""

import inspect
import logging
import typing
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia import enhance
from kornia.core import Tensor
from kornia.core.external import mcp
from kornia.io import load_image, write_image

logger = logging.getLogger(__name__)

EXCLUDED_ENHANCE_FUNCTIONS = [
    "histogram",
    "histogram2d",
    "image_histogram2d",
    "invert",
    "jpeg_codec_differentiable",
    "linear_transform",
    "normalize",
    "shift_rgb",
    "zca_mean",
    "zca_whiten",
]


def create_kornia_mcp_server() -> "mcp.server.fastmcp.FastMCP":
    """Create an MCP server for Kornia's modules."""
    mcp = mcp.server.fastmcp.FastMCP("Kornia")

    # Register functions from enhance module
    for name, func in inspect.getmembers(enhance, inspect.isfunction):
        if not name.startswith("_"):  # Skip private functions
            if name in EXCLUDED_ENHANCE_FUNCTIONS:
                continue
            # Create a wrapper function with proper type hints
            wrapper = _create_enhance_function_wrapper(func)
            # Register the wrapper as a tool
            mcp.add_tool(wrapper, name=f"kornia.enhance.{name}", description=func.__doc__)

    return mcp


def _create_enhance_function_wrapper(func: Any):
    """Create a wrapper function for an enhance function with proper type hints."""
    sig = inspect.signature(func)

    # Create parameter list with type hints
    params = []
    param_names = []
    for param_name, param in sig.parameters.items():
        if param.annotation == Tensor or param.annotation == "Tensor":
            if (
                param_name == "input"
                or param_name == "image"
                or param_name == "src1"
                or param_name == "src2"
                or param_name == "data"
                or param_name == "x"
            ):
                params.append(f"{param_name}_path: str")
                param_names.append(f"{param_name}_path")
            else:
                # an error will throw if the type is Tensor. I will keep it for now.
                params.append(f"{param_name}: Tensor")
                param_names.append(f"{param_name}")
        else:
            if type(param.annotation) == str:
                annotation = param.annotation
            else:
                annotation = "str" if param.annotation == inspect.Parameter.empty else param.annotation.__name__
            params.append(f"{param_name}: {annotation}")
            param_names.append(f"{param_name}")

    # Create the wrapper function with dynamic signature
    wrapper_code = f'''
def wrapper({", ".join(params)}) -> Dict[str, Any]:
    """Call the enhance function with proper argument handling."""
    kwargs = locals()

    # Process arguments
    processed_kwargs = {{}}

    # Process each argument
    for k, v in kwargs.items():
        if k.endswith('_path'):  # Handle image paths
            img = load_image(v)
            processed_kwargs[k.replace('_path', '')] = img
        else:
            processed_kwargs[k] = v

    # Call the original function
    result = func(**processed_kwargs)

    return {{'output': result}}
'''

    # Create namespace for the wrapper
    namespace = {
        "Dict": Dict,
        "Any": Any,
        "Union": Union,
        "Tuple": Tuple,
        "List": List,
        "Optional": Optional,
        "typing": typing,
        "load_image": load_image,
        "write_image": write_image,
        "torch": torch,
        "Tensor": Tensor,
        "func": func,
    }
    # Execute the wrapper code
    exec(wrapper_code, namespace)
    return namespace["wrapper"]
