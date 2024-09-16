from typing import Optional, ClassVar, List

import kornia
from kornia.core import Module, Tensor, ones_like, tensor, zeros_like
from kornia.core.external import segmentation_models_pytorch as smp
from kornia.core.module import ONNXExportMixin


class SegmentationModels(Module, ONNXExportMixin):
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

    ONNX_DEFAULT_INPUTSHAPE: ClassVar[List[int]] = (-1, 3, -1, -1)
    ONNX_DEFAULT_OUTPUTSHAPE: ClassVar[List[int]] = (-1, -1, -1, -1)

    def __init__(
        self,
        model_name: str = "Unet",
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.preproc_params = smp.encoders.get_preprocessing_params(encoder_name)  # type: ignore
        self.segmentation_model = getattr(smp, model_name)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs,
        )

    def preprocessing(self, input: Tensor) -> Tensor:
        # Ensure the color space transformation is ONNX-friendly
        input_space = self.preproc_params["input_space"]
        input = (
            kornia.color.rgb.rgb_to_bgr(input) if input_space == "BGR" else input
        )  # Assume input is already RGB if not BGR

        # Normalize input range if needed
        input_range = self.preproc_params["input_range"]
        if input_range[1] == 255:
            input = input * 255.0
        elif input_range[1] == 1:
            pass
        else:
            raise ValueError(f"Unsupported input range: {input_range}")

        # Handle mean and std normalization
        if self.preproc_params["mean"] is not None:
            mean = tensor([self.preproc_params["mean"]], device=input.device)
        else:
            mean = zeros_like(input)

        if self.preproc_params["std"] is not None:
            std = tensor([self.preproc_params["std"]], device=input.device)
        else:
            std = ones_like(input)

        return kornia.enhance.normalize(input, mean, std)

    def forward(self, input: Tensor) -> Tensor:
        input = self.preprocessing(input)
        return self.segmentation_model(input)
