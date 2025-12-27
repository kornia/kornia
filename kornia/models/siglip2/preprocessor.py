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

"""SigLip2 image preprocessor."""

from __future__ import annotations

from kornia.core import ImageSequential, Module, Tensor, tensor
from kornia.enhance.normalize import Normalize
from kornia.enhance.rescale import Rescale
from kornia.geometry.transform import Resize


class SigLip2ImagePreprocessor(Module):
    """Image preprocessor for SigLip2 models.

    This preprocessor matches the HuggingFace SiglipImageProcessor behavior:
    - Rescales pixel values from [0, 255] to [0, 1]
    - Resizes images to the specified size (with bicubic interpolation and antialiasing)
    - Normalizes with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5] (converts [0, 1] to [-1, 1])

    Note:
        Small differences from HuggingFace preprocessing are expected due to interpolation
        implementation differences between PIL (used by HF) and PyTorch (used by Kornia).
        These differences are within acceptable tolerance and do not significantly affect
        model performance.

    Args:
        image_size: Target image size (height, width). Default: (224, 224)
        mean: Normalization mean. Default: [0.5, 0.5, 0.5]
        std: Normalization std. Default: [0.5, 0.5, 0.5]
        rescale_factor: Rescaling factor. Default: 1/255

    Example:
        >>> import torch
        >>> from kornia.models.siglip2 import SigLip2ImagePreprocessor
        >>>
        >>> # Create preprocessor
        >>> preprocessor = SigLip2ImagePreprocessor(image_size=(224, 224))
        >>>
        >>> # Process image (assumes input in [0, 255] range)
        >>> image = torch.randint(0, 255, (3, 300, 400), dtype=torch.float32)
        >>> processed = preprocessor(image)  # Shape: (3, 224, 224), range: [-1, 1]
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        mean: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
        rescale_factor: float = 1.0 / 255.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.mean = tensor([mean]) if isinstance(mean, (list, tuple)) else mean
        self.std = tensor([std]) if isinstance(std, (list, tuple)) else std
        self.rescale_factor = rescale_factor

        # Build preprocessing pipeline
        # Order matches HF: Rescale -> Resize -> Normalize
        # This ensures resize operates on [0, 1] range, avoiding out-of-range values
        preproc_list: list[Module] = []

        # Rescale first (convert [0, 255] to [0, 1])
        # This should be done before resize to match HF behavior
        if rescale_factor != 1.0:
            preproc_list.append(Rescale(factor=rescale_factor))

        # Resize (on [0, 1] range)
        # Use antialias=True to better match PIL's resize behavior
        preproc_list.append(Resize(size=image_size, interpolation="bicubic", align_corners=False, antialias=True))

        # Normalize (convert [0, 1] to [-1, 1])
        preproc_list.append(Normalize(mean=self.mean, std=self.std))

        self.preprocessor = ImageSequential(*preproc_list)

    def forward(self, images: Tensor) -> Tensor:
        """Preprocess images.

        Args:
            images: Input images. Can be:
                - Single image: (C, H, W) or (1, C, H, W)
                - Batch of images: (B, C, H, W)
                - Images in [0, 255] range (will be rescaled)

        Returns:
            Preprocessed images in shape (B, C, H, W) with values in [-1, 1] range.
        """
        # Ensure batch dimension
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Process through pipeline
        return self.preprocessor(images)

    @classmethod
    def from_config(cls, image_size: int | tuple[int, int]) -> SigLip2ImagePreprocessor:
        """Create preprocessor from image size configuration.

        Args:
            image_size: Image size (single int for square, or tuple for (height, width))

        Returns:
            Preprocessor instance
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return cls(image_size=image_size)
