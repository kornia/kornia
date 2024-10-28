from __future__ import annotations

from typing import Any, Optional

from torch import nn

import kornia
from kornia.core import Module, tensor
from kornia.core.external import segmentation_models_pytorch as smp

from .base import SemanticSegmentation

__all__ = ["SegmentationModelsBuilder"]


class SegmentationModelsBuilder:
    @staticmethod
    def build(
        model_name: str = "Unet",
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: str = "softmax",
        **kwargs: Any,
    ) -> SemanticSegmentation:
        """SegmentationModel is a module that wraps a segmentation model.

        This module uses SegmentationModel library for segmentation.

        Args:
            model_name: Name of the model to use. Valid options are:
                "Unet", "UnetPlusPlus", "MAnet", "LinkNet", "FPN", "PSPNet", "PAN", "DeepLabV3", "DeepLabV3Plus".
            encoder_name: Name of the encoder to use.
            encoder_depth: Depth of the encoder.
            encoder_weights: Weights of the encoder.
            decoder_channels: Number of channels in the decoder.
            in_channels: Number of channels in the input.
            classes: Number of classes to predict.
            **kwargs: Additional arguments to pass to the model. Detailed arguments can be found at:
                https://github.com/qubvel-org/segmentation_models.pytorch/tree/main/segmentation_models_pytorch/decoders

        Note:
            Only encoder weights are available.
            Pretrained weights for the whole model are not available.
        """

        preproc_params = smp.encoders.get_preprocessing_params(encoder_name)  # type: ignore
        preprocessor = SegmentationModelsBuilder.get_preprocessing_pipeline(preproc_params)
        segmentation_model = getattr(smp, model_name)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            **kwargs,
        )

        return SemanticSegmentation(
            model=segmentation_model,
            pre_processor=preprocessor,
            post_processor=nn.Identity(),
            name=f"{model_name}_{encoder_name}",
        )

    @staticmethod
    def get_preprocessing_pipeline(preproc_params: dict[str, Any]) -> kornia.augmentation.container.ImageSequential:
        # Ensure the color space transformation is ONNX-friendly
        proc_sequence: list[Module] = []
        input_space = preproc_params["input_space"]
        if input_space == "BGR":
            proc_sequence.append(kornia.color.BgrToRgb())
        elif input_space == "RGB":
            pass
        else:
            raise ValueError(f"Unsupported input space: {input_space}")

        # Normalize input range if needed
        input_range = preproc_params["input_range"]
        if input_range[1] == 255:
            proc_sequence.append(kornia.enhance.Normalize(mean=0.0, std=1 / 255.0))
        elif input_range[1] == 1:
            pass
        else:
            raise ValueError(f"Unsupported input range: {input_range}")

        # Handle mean and std normalization
        if preproc_params["mean"] is not None:
            mean = tensor([preproc_params["mean"]])
        else:
            mean = tensor(0.0)

        if preproc_params["std"] is not None:
            std = tensor([preproc_params["std"]])
        else:
            std = tensor(1.0)
        proc_sequence.append(kornia.enhance.Normalize(mean=mean, std=std))

        return kornia.augmentation.container.ImageSequential(*proc_sequence)
