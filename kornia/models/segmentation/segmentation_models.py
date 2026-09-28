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

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

import kornia
from kornia.core.check import KORNIA_CHECK

from .base import SemanticSegmentation

__all__ = ["SegmentationModelsBuilder"]

_PREPROC_KEYS = ("input_space", "input_range", "mean", "std")


class SegmentationModelsBuilder:
    """Wrap a segmentation network and its encoder's preprocessing in a :class:`SemanticSegmentation`.

    The builder is written for networks from `segmentation_models_pytorch
    <https://github.com/qubvel-org/segmentation_models.pytorch>`_ (smp), whose encoders ship the
    preprocessing parameters their pretrained weights expect, but any ``nn.Module`` mapping a
    ``(B, 3, H, W)`` image batch to a ``(B, C, H, W)`` prediction works. Kornia does not import smp:
    you build the network and fetch its preprocessing parameters, and the builder supplies the
    ONNX-friendly preprocessing pipeline and the container. Give the network a softmax head
    (``activation="softmax2d"`` in smp) if you want :meth:`SemanticSegmentation.visualize`: it
    expects per-pixel class probabilities and raises on raw logits.

    Example:
        >>> import segmentation_models_pytorch as smp  # doctest: +SKIP
        >>> from kornia.models.segmentation import SegmentationModelsBuilder
        >>> net = smp.Unet(
        ...     encoder_name="resnet34", encoder_weights="imagenet", classes=2, activation="softmax2d"
        ... )  # doctest: +SKIP
        >>> params = smp.encoders.get_preprocessing_params("resnet34")  # doctest: +SKIP
        >>> model = SegmentationModelsBuilder.build(net, params, name="Unet_resnet34")  # doctest: +SKIP
        >>> model(torch.rand(1, 3, 64, 64)).shape  # doctest: +SKIP
        torch.Size([1, 2, 64, 64])
    """

    @staticmethod
    def build(
        model: nn.Module,
        preproc_params: Optional[dict[str, Any]] = None,
        name: str = "segmentation_model",
    ) -> SemanticSegmentation:
        """Wrap a constructed segmentation network in a :class:`SemanticSegmentation`.

        Args:
            model: The segmentation network, e.g. ``smp.Unet(...)``. It is put in ``eval`` mode.
            preproc_params: The preprocessing parameters of the network's encoder, in the shape
                returned by ``smp.encoders.get_preprocessing_params(encoder_name)``: the keys
                ``input_space`` (``"RGB"`` or ``"BGR"``), ``input_range`` (``[0, 1]`` or ``[0, 255]``),
                ``mean`` and ``std`` (per-channel lists, or ``None`` for no normalization). See
                :meth:`get_preprocessing_pipeline`. ``None`` feeds the input to the network unchanged.
            name: Name of the wrapped model; :meth:`SemanticSegmentation.save` uses it for file names.

        Returns:
            The container running preprocessing, the network and an identity post-processor.

        """
        if preproc_params is None:
            preprocessor: nn.Module = nn.Identity()
        else:
            preprocessor = SegmentationModelsBuilder.get_preprocessing_pipeline(preproc_params)

        return SemanticSegmentation(
            model=model,
            pre_processor=preprocessor,
            post_processor=nn.Identity(),
            name=name,
        )

    @staticmethod
    def get_preprocessing_pipeline(preproc_params: dict[str, Any]) -> kornia.augmentation.container.ImageSequential:
        """Build the preprocessing pipeline expected by a segmentation model.

        Args:
            preproc_params: Dictionary from the segmentation-model metadata, e.g.
                ``smp.encoders.get_preprocessing_params(encoder_name)``. It must carry the keys
                ``input_space`` (``"RGB"`` or ``"BGR"``: the color order the network was trained on,
                so a ``"BGR"`` network gets its RGB input flipped), ``input_range`` (``[0, 1]`` or
                ``[0, 255]``: the range the ``mean``/``std`` are expressed in, so ``[0, 255]``
                multiplies the ``[0, 1]`` input by 255 first), and ``mean`` and ``std`` (per-channel
                lists, or ``None`` for no normalization).

        Returns:
            :class:`~kornia.augmentation.container.ImageSequential` containing
            ONNX-friendly color conversion, rescaling, and normalization steps.

        Note:
            Set ``pipeline.disable_features = True`` before exporting the returned
            pipeline to ONNX. This disables convenience input/output conversion and
            output caching, whose tensor attribute mutation is rejected by some
            versions of ``torch.export``.

        Raises:
            BaseError: If one of the four keys is missing (a :func:`~kornia.core.check.KORNIA_CHECK`).
            ValueError: If ``input_space`` or ``input_range`` is not one of the supported values.

        """
        for key in _PREPROC_KEYS:
            KORNIA_CHECK(key in preproc_params, f"preproc_params is missing the key '{key}'")

        # Ensure the color space transformation is ONNX-friendly
        proc_sequence: list[nn.Module] = []
        input_space = preproc_params["input_space"]
        if input_space == "BGR":
            proc_sequence.append(kornia.color.BgrToRgb())
        elif input_space == "RGB":
            pass
        else:
            raise ValueError(f"Unsupported input space: {input_space}")

        # Rescale the [0, 1] input to [0, 255] if the network expects it. Multiply by 255, which every
        # dtype stores exactly, rather than divide by the reciprocal: bfloat16 rounds 1/255 to 0.0039368,
        # a ~254.0 multiplier that maps 0.5 to 127.0 instead of 127.5. `Rescale` holds the factor as a
        # 0-d tensor and exports to ONNX from any device.
        input_range = preproc_params["input_range"]
        if input_range[1] == 255:
            proc_sequence.append(kornia.enhance.Rescale(255.0))
        elif input_range[1] == 1:
            pass
        else:
            raise ValueError(f"Unsupported input range: {input_range}")

        # Handle mean and std normalization
        if preproc_params["mean"] is not None:
            mean = torch.tensor([preproc_params["mean"]])
        else:
            mean = torch.tensor(0.0)

        if preproc_params["std"] is not None:
            std = torch.tensor([preproc_params["std"]])
        else:
            std = torch.tensor(1.0)
        proc_sequence.append(kornia.enhance.Normalize(mean=mean, std=std))

        return kornia.augmentation.container.ImageSequential(*proc_sequence)
