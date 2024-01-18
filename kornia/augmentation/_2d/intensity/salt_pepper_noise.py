from typing import Any, Dict, Optional, Tuple, Union

import torch

from kornia.augmentation import random_generator as rg
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK


class RandomSaltAndPepperNoise(IntensityAugmentationBase2D):
    """Apply salt-and-pepper noise to the input image.

    Args:
        amount (float or Tuple[float, float]): Range of noise values to be added. Default is (0.01, 0.06).
            The valid range is [0, 1], where 0 represents no noise and 1 represents full noise.
        salt_vs_pepper (float or Tuple[float, float]): Range of salt vs. pepper ratio. Default is (0.4, 0.6).
            The valid range is [0, 1], where 0 represents only pepper noise, 1 represents only salt noise,
            and 0.5 represents an equal mix.
        p (float): Probability of applying the transformation. Default is 0.5.
        same_on_batch (bool): Apply the same transformation across the batch. Default is False.
        keepdim (bool): Keep the same dimensions after transformation. Default is False.

    Attributes:
        _param_generator (PlainUniformGenerator): Parameter generator for random parameter generation.

    Raises:
        ValueError: If `salt_vs_pepper` or `amount` are not in the valid range.

    Note:
        The `amount` and `salt_vs_pepper` parameters control the intensity of the salt-and-pepper noise.
        `amount` determines the range of noise values to be added, and `salt_vs_pepper` determines the
        range of the salt vs. pepper ratio.

    Example:
        >>> transform = RandomSaltAndPepperNoise(amount=(0.02, 0.1), salt_vs_pepper=(0.3, 0.7), p=1.0)
        >>> noisy_image = transform(input_image)
    """

    def __init__(
        self,
        amount: Optional[Union[float, Tuple[float, float]]] = (0.01, 0.06),
        salt_vs_pepper: Optional[Union[float, Tuple[float, float]]] = (0.4, 0.6),
        p: float = 0.5,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

        if isinstance(salt_vs_pepper, (tuple, float)):
            if isinstance(salt_vs_pepper, float):
                salt_vs_pepper = (salt_vs_pepper, salt_vs_pepper)
            elif len(salt_vs_pepper) == 1:
                salt_vs_pepper = (salt_vs_pepper[0], salt_vs_pepper[0])
            elif len(salt_vs_pepper) > 2 or len(salt_vs_pepper) <= 0:
                raise ValueError(
                    "The length of salt_vs_pepper must be greater than 0 \
                        and less than or equal to 2, and it should be a tuple."
                )
        else:
            raise ValueError("salt_vs_pepper must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in salt_vs_pepper),
            "Salt_vs_pepper values must be between 0 and 1. \
                        Recommended value 0.5.",
        )

        if isinstance(amount, (tuple, float)):
            if isinstance(amount, float):
                amount = (amount, amount)
            elif len(amount) == 1:
                amount = (amount[0], amount[0])
            elif len(amount) > 2 or len(amount) <= 0:
                raise ValueError(
                    "The length of amount must be greater than 0 \
                        and less than or equal to 2, and it should be a tuple."
                )
        else:
            raise ValueError("amount must be a tuple or a float")
        KORNIA_CHECK(
            all(0 <= el <= 1 for el in amount),
            "amount of noise values must be between 0 and 1. \
                        Recommended values less than 0.2.",
        )

        self._param_generator = rg.PlainUniformGenerator(
            (amount, "amount", None, None), (salt_vs_pepper, "salt_vs_pepper", None, None)
        )

    def _generate_params(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate salt and pepper noise masks.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tuple of salt and pepper masks.
        """
        B, C, H, W = input.size()
        _dev = input.device
        mask_noise = torch.rand(B, 1, H, W, device=_dev) < self._params["amount"].view(B, 1, 1, 1).to(_dev)
        mask_salt = torch.rand(B, 1, H, W, device=_dev) < self._params["salt_vs_pepper"].view(B, 1, 1, 1).to(_dev)
        if C > 1:
            mask_noise = mask_noise.repeat(1, C, 1, 1)
            mask_salt = mask_salt.repeat(1, C, 1, 1)
        mask_pepper = (~(~mask_salt & mask_noise)).float()
        mask_salt = (mask_salt & mask_noise).float()
        self._params["salt_and_pepper_noise"] = [mask_salt, mask_pepper]

        return mask_salt, mask_pepper

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply salt-and-pepper noise transformation to the input.

        Args:
            input (Tensor): Input tensor.
            params (Dict[str, Tensor]): Transformation parameters.
            flags (Dict[str, Any]): Transformation flags.
            transform (Optional[Tensor]): Additional transformation.

        Returns:
            Tensor: Transformed input tensor.
        """

        # Create noisy image cloning input.
        noisy_image = input.clone()

        # Generate noisy masks or read from params.
        if "salt_and_pepper_noise" in params:
            mask_salt = params["salt_and_pepper_noise"][0]
            mask_pepper = params["salt_and_pepper_noise"][1]
        else:
            mask_salt, mask_pepper = self._generate_params(input)
            self._params["salt_and_pepper_noise"] = [mask_salt, mask_pepper]

        # Apply add and multiply mask.
        noisy_image += mask_salt
        noisy_image *= mask_pepper

        return torch.clamp(noisy_image, 0, 1)
