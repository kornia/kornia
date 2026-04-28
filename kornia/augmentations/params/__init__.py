"""kornia.augmentations.params — Pydantic models describing transform parameters.

Currently scopes the seven rf-detr-critical transforms (PR-PV).
Mass migration to all transforms ships in a follow-up.
"""
from kornia.augmentations.params.rfdetr_seven import PARAMS_BY_NAME, PYDANTIC_AVAILABLE

if PYDANTIC_AVAILABLE:
    from kornia.augmentations.params.rfdetr_seven import (
        ColorJiggleParams,
        RandomAffineParams,
        RandomGaussianBlurParams,
        RandomGaussianNoiseParams,
        RandomHorizontalFlipParams,
        RandomRotationParams,
        RandomVerticalFlipParams,
    )

__all__ = ["PARAMS_BY_NAME", "PYDANTIC_AVAILABLE"]
