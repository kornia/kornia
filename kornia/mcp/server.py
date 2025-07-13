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
import itertools
import logging
import typing
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from kornia.core import Tensor
from kornia.core.external import mcp as _mcp
from kornia.mcp.utils import (
    load_any_image,
    tensor_to_base64,
    parse_args_from_docstring,
    parse_description_from_docstring,
)

logger = logging.getLogger(__name__)

shared_namespace = {
    "Dict": Dict,
    "Any": Any,
    "Union": Union,
    "Tuple": Tuple,
    "List": List,
    "Optional": Optional,
    "typing": typing,
    "load_any_image": load_any_image,
    "tensor_to_base64": tensor_to_base64,
    "torch": torch,
    "Tensor": Tensor,
    "logger": logger,
}


def add_func_as_tool(
    mcp: "_mcp.server.fastmcp.FastMCP", func: Any, tool_prefix: str
) -> "_mcp.server.fastmcp.FastMCP":
    """Add a function as a tool to the MCP server."""
    wrapper = _create_function_wrapper(func)
    description = parse_description_from_docstring(func.__doc__)
    mcp.add_tool(
        wrapper,
        name=f"{tool_prefix}_{func.__name__}",
        description=description,
    )
    return mcp

    
def add_class_as_tool(
    mcp: "_mcp.server.fastmcp.FastMCP", cls: Any, tool_prefix: str
) -> "_mcp.server.fastmcp.FastMCP":
    """Add a class as a tool to the MCP server."""
    wrapper = _create_class_wrapper(cls)
    description = parse_description_from_docstring(cls.__doc__)
    mcp.add_tool(
        wrapper,
        name=f"{tool_prefix}_{cls.__name__}",
        description=description,
    )
    return mcp


def _create_function_wrapper(func: Any):
    """Create a wrapper function for an enhance function with proper type hints."""
    sig = inspect.signature(func)

    if not (sig.return_annotation == "Tensor" or sig.return_annotation == Tensor):
        raise ValueError(f"Currently, the function must return a Tensor. Obtained {sig.return_annotation}.")

    # Create parameter list with type hints
    params = []
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
                params.append(f"{param_name}_uri: Any")
            # an error will throw if the type is Tensor. I will keep it for now.
            elif param.default != inspect.Parameter.empty:
                params.append(f"{param_name}: Any = {param.default}")
            else:
                params.append(f"{param_name}: Any")
        else:
            if type(param.annotation) == str:
                annotation = param.annotation
            else:
                annotation = "str" if param.annotation == inspect.Parameter.empty else param.annotation.__name__
            if param.default != inspect.Parameter.empty:
                params.append(f"{param_name}: {annotation} = {param.default}")
            else:
                params.append(f"{param_name}: {annotation}")

    # Create the wrapper function with dynamic signature
    wrapper_code = f'''
def wrapper({", ".join(params)}, return_type: str = "base64"):
    """Call the enhance function with proper argument handling."""
    kwargs = locals()

    # Process arguments
    processed_kwargs = {{}}

    # Process each argument
    for k, v in kwargs.items():
        if k.endswith('_uri'):  # Handle image paths
            img = load_any_image(v) / 255.0
            processed_kwargs[k.replace('_uri', '')] = img
        elif k == "return_type":
            continue
        else:
            processed_kwargs[k] = v

    # Call the original function
    result = func(**processed_kwargs)

    if return_type == "numpy":
        return result.cpu().numpy()
    elif return_type == "torch":
        return result
    elif return_type == "base64":
        return tensor_to_base64(result)
    raise ValueError(f"Invalid return type: {{return_type}}")
'''
    namespace = {**shared_namespace, "func": func}
    # Execute the wrapper code
    exec(wrapper_code, namespace)
    return namespace["wrapper"]


def _create_class_wrapper(cls: Any):
    """Create a wrapper function for an enhance class with proper type hints."""
    # Get signatures for both constructor and forward method
    init_sig = inspect.signature(cls)
    forward_sig = inspect.signature(cls.forward)

    if not (forward_sig.return_annotation == "Tensor" or forward_sig.return_annotation == Tensor):
        raise ValueError(
            f"Currently, the forward method must return a Tensor. Obtained {forward_sig.return_annotation}."
        )

    # Create parameter list with type hints
    params = []
    params_with_default = []

    for param_name, param in forward_sig.parameters.items():
        if param_name == "self":
            continue
        if param.annotation == Tensor or param.annotation == "Tensor":
            if (
                param_name == "input"
                or param_name == "image"
                or param_name == "src1"
                or param_name == "src2"
                or param_name == "data"
                or param_name == "x"
            ):
                params.append(f"{param_name}_uri: Any")
            # an error will throw if the type is Tensor. I will change it to Any for now.
            elif param.default != inspect.Parameter.empty:
                params_with_default.append(f"{param_name}: Any = {param.default}")
            else:
                params.append(f"{param_name}: Any")
        else:
            if type(param.annotation) == str:
                annotation = param.annotation
            else:
                annotation = "str" if param.annotation == inspect.Parameter.empty else param.annotation.__name__
            params.append(f"{param_name}: {annotation}")

    for param_name, param in init_sig.parameters.items():
        if param_name == "self":
            continue
        if type(param.annotation) == str:
            annotation = param.annotation
        else:
            annotation = "str" if param.annotation == inspect.Parameter.empty else param.annotation.__name__
        if param.default != inspect.Parameter.empty:
            if annotation == "str":
                params_with_default.append(f"{param_name}: {annotation} = '{param.default}'")
            else:
                params_with_default.append(f"{param_name}: {annotation} = {param.default}")
        else:
            params.append(f"{param_name}: {annotation}")

    # Create the wrapper function with dynamic signature
    wrapper_code = f'''
def wrapper({", ".join(itertools.chain(params, params_with_default))}, return_type: str = "base64"):
    """Call the enhance class with proper argument handling."""
    kwargs = locals()

    # Process arguments
    processed_kwargs = {{}}

    # Process each argument
    for k, v in kwargs.items():
        if k.endswith('_uri'):  # Handle image paths
            img = load_any_image(v) / 255.0
            processed_kwargs[k.replace('_uri', '')] = img
        elif k == "return_type":
            continue
        else:
            processed_kwargs[k] = v

    init_arguments = [{", ".join(["'" + str(k) + "'" for k in init_sig.parameters])}]
    # Create an instance of the class
    instance = cls(**{{k: processed_kwargs[k] for k in init_arguments}})

    # Call the forward method
    result = instance(**{{
        k: processed_kwargs[k] for k in processed_kwargs if k not in init_arguments}})

    if return_type == "numpy":
        return result.cpu().numpy()
    elif return_type == "torch":
        return result
    elif return_type == "base64":
        return tensor_to_base64(result)
    raise ValueError(f"Invalid return type: {{return_type}}")
'''
    # Create namespace for the wrapper
    namespace = {**shared_namespace, "cls": cls}
    # Execute the wrapper code
    exec(wrapper_code, namespace)
    return namespace["wrapper"]
