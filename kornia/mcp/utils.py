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

import base64
import io
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Union

from kornia.core import Tensor
from kornia.core.external import PILImage as Image
from kornia.core.external import numpy as np
from kornia.io import load_image
from kornia.utils import image_to_tensor


def tensor_to_base64(image: Tensor) -> str:
    image = image.permute(1, 2, 0)
    pil_img = (
        Image.fromarray((image.cpu().numpy() * 255).astype("uint8"))
        if image.max() <= 1.0
        else Image.fromarray(image.cpu().numpy().astype("uint8"))
    )
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_tensor(base64_str: str) -> Tensor:
    img_bytes = base64.b64decode(base64_str.split(",")[1])
    img_pil = Image.open(io.BytesIO(img_bytes))
    img_np = np.array(img_pil)
    return image_to_tensor(img_np)


def load_any_image(image: Union[str, Path, "np.ndarray", Tensor]) -> Tensor:
    """Load input image from various sources into a torch tensor.

    Args:
        image: Can be one of:
            - String or Path filepath to local image
            - URL string to remote image
            - Base64 encoded image string
            - Numpy array of shape (H,W), (H,W,C)
            - Torch tensor of shape (H,W), (H,W,C), (C,H,W)

    Returns:
        Tensor: Image tensor with shape (C,H,W)

    Example:
        >>> # Load from filepath
        >>> img = load_input_image('path/to/image.jpg')
        >>> # Load from URL
        >>> img = load_input_image('https://example.com/image.jpg')
        >>> # Load from base64 string
        >>> img = load_input_image('data:image/jpeg;base64,/9j/4AAQSkZJRg...')
        >>> # Load from numpy array
        >>> img = load_input_image(np.random.rand(100,100,3))
        >>> # Load from tensor
        >>> img = load_input_image(torch.randn(3,100,100))
    """
    if isinstance(image, (str, Path)):
        if str(image).startswith(("http://", "https://")):
            # Download image from URL to temp file
            temp_file, _ = urllib.request.urlretrieve(str(image))
            img = load_image(temp_file)
            os.unlink(temp_file)  # Clean up temp file
            return img
        elif str(image).startswith("data:image"):
            # Handle base64 encoded image
            import base64
            import io

            from PIL import Image

            # Extract the actual base64 string after the comma
            base64_str = str(image).split(",")[1]
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_str)
            # Convert to PIL Image
            img_pil = Image.open(io.BytesIO(img_bytes))
            # Convert to numpy array
            img_np = np.array(img_pil)
            return image_to_tensor(img_np)
        return load_image(image)

    if isinstance(image, np.ndarray):
        return image_to_tensor(image)

    if isinstance(image, Tensor):
        # Handle different tensor shapes
        if len(image.shape) == 2:
            return image.unsqueeze(0)
        if len(image.shape) == 3:
            if image.shape[0] in [1, 3]:  # Already in CHW format
                return image
            return image.permute(2, 0, 1)  # Convert HWC to CHW
        raise ValueError(f"Invalid tensor shape {image.shape}. Expected 2D or 3D tensor.")

    raise TypeError(f"Unsupported input type: {type(image)}")


def parse_args_from_docstring(docstring: str) -> List[Dict[str, dict]]:
    """Parses a Google-style docstring's Args section (without types)
    and returns a dictionary input schema.
    """
    parameters = []

    # Match Args section and stop at next section like "Shape:", "Example:", etc.
    match = re.search(r"Args:\s*((?:\n {4}.+?:.*(?:\n {6}.*)*))", docstring)
    if not match:
        return parameters

    args_block = match.group(1)

    # Match each param block (name: description [multi-line supported])
    param_matches = re.findall(r" {4}(\w+): (.*?)(?=(?:\n {4}\w+:|\Z))", args_block, re.DOTALL)

    for name, desc in param_matches:
        # Collapse multiline descriptions into one line
        full_desc = " ".join(line.strip() for line in desc.strip().splitlines())

        parameters.append(
            {
                "type": "object",
                "properties": {name: {"type": "string", "description": full_desc}},
            }
        )
    return parameters


def parse_description_from_docstring(docstring: str) -> str:
    """Extracts the general description from a Google-style docstring,
    stopping just before the Args: section or other section headers (Shape, Example, etc.)
    """
    if not docstring:
        return ""

    # Split docstring into sections
    lines = docstring.strip().splitlines()
    desc_lines = []
    for line in lines:
        # Stop at any section header like 'Returns:', 'Shape:', etc.
        if re.match(r"^\s*(Returns|Shape|Example|Examples|Note|Notes):", line):
            break
        desc_lines.append(line.rstrip())

    return "\n".join(desc_lines).strip()
