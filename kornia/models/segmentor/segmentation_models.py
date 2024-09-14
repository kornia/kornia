from typing import Optional

import kornia
from kornia.core import Module, Tensor, tensor
from kornia.core.external import segmentation_models_pytorch as smp


class SegmentationModels(Module):
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
    def __init__(
        self,
        model_name: str = "Unet",
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        **kwargs
    ) -> None:
        super().__init__()
        self.preproc_params = smp.encoders.get_preprocessing_params(encoder_name)  # type: ignore
        self.segmentation_model = getattr(smp, model_name)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs
        )

    def preprocessing(self, input: Tensor) -> Tensor:
        if self.preproc_params["input_space"] == "RGB":
            pass
        elif self.preproc_params["input_space"] == "BGR":
            input = kornia.color.rgb_to_bgr(input)
        else:
            raise ValueError(f"Unsupported input space: {self.preproc_params['input_space']}")
        
        if self.preproc_params["input_range"] is not None:
            if input.max() > 1 and self.preproc_params["input_range"][1] == 1:
                input = input / 255.0

        if self.preproc_params["mean"] is None:
            mean = tensor(self.preproc_params["mean"]).to(input.device)
        else:
            mean = tensor(self.preproc_params["mean"]).to(input.device)
        
        if self.preproc_params["std"] is None:
            std = tensor(self.preproc_params["std"]).to(input.device)
        else:
            std = tensor(self.preproc_params["std"]).to(input.device)

        return kornia.enhance.normalize(input, mean, std)

    def forward(self, input: Tensor) -> Tensor:
        input = self.preprocessing(input)
        return self.segmentation_model(input)
