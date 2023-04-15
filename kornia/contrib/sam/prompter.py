from __future__ import annotations

from torch import device

from kornia.augmentation import LongestMaxSize
from kornia.contrib import Sam, SamModelType
from kornia.contrib.models.prompters.image import ImagePrompter
from kornia.core import Tensor, pad, tensor
from kornia.enhance import normalize


class SamPrompter(ImagePrompter):
    """This class allow the user to run multiple query with multiple prompts for a SAM model.

    A SAM model is loaded based on the given parameters and provides this to a `ImagePrompter`.

    For default the images are transformed to have their long side with size of the `image_encoder.img_size`. This
    Prompter class ensure to transform the images and the prompts before prediction. Also, the image is passed
    automatically for the method `preprocess_image`, which is responsible for normalize the image and pad it to have
    the right size for the SAM model :math:`(\text{image_encoder.img_size}, \text{image_encoder.img_size})`. For
    default the image is normalized by the mean and standard deviation of the SAM dataset values.

    Args:
        model_type: the available models are:

            - 0, 'vit_h' or :func:`kornia.contrib.sam.SamModelType.vit_h`
            - 1, 'vit_l' or :func:`kornia.contrib.sam.SamModelType.vit_l`
            - 2, 'vit_b' or :func:`kornia.contrib.sam.SamModelType.vit_b`

        checkpoint: The url or filepath for the respective checkpoint
        device: The desired device to load the weights and move the model
    """

    def __init__(
        self,
        model_type: str | int | SamModelType = SamModelType.vit_b,
        checkpoint: str | None = None,
        device: device | None = None,
    ) -> None:
        model = Sam.from_pretrained(model_type, checkpoint=checkpoint, device=device)

        super().__init__(model, LongestMaxSize(model.image_encoder.img_size, p=1.0))

        self.pixel_mean: Tensor | None = tensor([123.675, 116.28, 103.53], device=device)
        self.pixel_std: Tensor | None = tensor([58.395, 57.12, 57.375], device=device)

    def preprocess_image(self, x: Tensor, mean: Tensor | None = None, std: Tensor | None = None) -> Tensor:
        """Normalize and pad a tensor.

        For normalize the tensor: will priorize the `mean` and `std` passed as argument, if None will use the default
        Sam Dataset values.

        For pad the tensor: Will pad the tensor into the right and bottom to match with the size of
        `self.model.image_encoder.img_size`

        Args:
            x: The image to be preprocessed
            mean: Mean for each channel.
            std: Standard deviations for each channel.

        Returns:
            The image preprocessed (normalized if has mean and str available and padded to encoder size)
        """

        if isinstance(mean, Tensor) and isinstance(std, Tensor):
            x = normalize(x, mean, std)
        elif isinstance(self.pixel_mean, Tensor) and isinstance(self.pixel_std, Tensor):
            x = normalize(x, self.pixel_mean, self.pixel_std)

        encoder_im_size = self.model.image_encoder.img_size
        pad_h = encoder_im_size - x.shape[-2]
        pad_w = encoder_im_size - x.shape[-1]
        x = pad(x, (0, pad_w, 0, pad_h))

        return x
