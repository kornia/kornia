from typing import Any, Dict, Optional, Tuple, Union

from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.augmentation.random_generator._2d import SaltAndPepperGenerator
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK


class RandomSaltAndPepperNoise(IntensityAugmentationBase2D):
    r"""Apply random Salt and Pepper noise to input images.

    .. image:: _static/img/RandomSaltAndPepperNoise.png

    Args:
        amount: A float or a tuple representing the range for the amount of noise to apply.
        salt_vs_pepper: A float or a tuple representing the range for the ratio of Salt to Pepper noise.
        p: The probability of applying the transformation. Default is 0.5.
        same_on_batch: If True, apply the same transformation across the entire batch. Default is False.
        keepdim: whether to keep the output shape the same as input (True) or broadcast it
                to the batch form (False).

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        The `amount` parameter controls the intensity of the noise, while `salt_vs_pepper` controls the ratio
        of Salt to Pepper noise.

        The values for `amount` and `salt_vs_pepper` should be between 0 and 1. The recommended value for
        `salt_vs_pepper` is 0.5, and for `amount`, values less than 0.2 are recommended.

        If `amount` and `salt_vs_pepper` are floats (unique values), the transformation is applied with these
        exact values, rather than randomly sampling from the specified range. However, the masks are still
        generated randomly using these exact parameters.

    Examples:
        >>> rng = torch.manual_seed(5)
        >>> inputs = torch.rand(1, 3, 3, 3)
        >>> aug = RandomSaltAndPepperNoise(amount=0.5, salt_vs_pepper=0.5, p=1.)
        >>> aug(inputs)
        tensor([[[[1.0000, 0.0000, 0.0000],
                  [1.0000, 1.0000, 0.1166],
                  [0.1644, 0.7379, 0.0000]],
        <BLANKLINE>
                 [[1.0000, 0.0000, 0.0000],
                  [1.0000, 1.0000, 0.7150],
                  [0.5793, 0.9809, 0.0000]],
        <BLANKLINE>
                 [[1.0000, 0.0000, 0.0000],
                  [1.0000, 1.0000, 0.7850],
                  [0.9752, 0.0903, 0.0000]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.rand(1, 3, 32, 32)
        >>> aug = RandomSaltAndPepperNoise(amount=0.05, salt_vs_pepper=0.5, p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
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

        # Validation and initialization of amount and salt_vs_pepper parameters.
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

        # Generator of random parameters and masks.
        self._param_generator = SaltAndPepperGenerator(amount, salt_vs_pepper)

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Apply random Salt and Pepper noise transformation to the input image."""
        KORNIA_CHECK(len(input.shape) in (3, 4), "Wrong input dimension.")
        if len(input.shape) == 3:
            input = input[None, :, :, :]
        KORNIA_CHECK(input.shape[1] in {3, 1}, "Number of color channels should be 1 or 3.")

        noisy_image = input.clone()

        # Apply noise masks using indexing.
        noisy_image[params["mask_salt"].to(input.device)] = 1.0
        noisy_image[params["mask_pepper"].to(input.device)] = 0.0

        return noisy_image
