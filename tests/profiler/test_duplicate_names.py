import pytest

from kornia.profiler.engines.augment import AugmentationPipeline


def test_duplicate_augmentation_names_raise_error() -> None:
    """Reject duplicate augmentation names in the configuration."""
    config = [
        {"name": "RandomRotation", "degrees": 15.0, "p": 1.0},
        {"name": "RandomRotation", "degrees": 30.0, "p": 1.0},
    ]

    with pytest.raises(ValueError, match="Augmentation names must be unique"):
        AugmentationPipeline(augmentations_config=config, mode="individual")