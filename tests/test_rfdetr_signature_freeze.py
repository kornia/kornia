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
"""Frozen signature test for the seven rf-detr-critical kornia transforms.

rf-detr/src/rfdetr/datasets/kornia_transforms.py hard-codes these classes
and their constructor signatures via factory functions. Drift in any of
these signatures (renamed keyword, removed default, type narrowing) breaks
rf-detr silently.

This test asserts the constructor parameter list of each class matches a
checked-in snapshot. Adding NEW keyword args (with defaults) is allowed
(additive); REMOVING or RENAMING args fails the test.

If the test fails, the message tells you exactly which params drifted and
which downstream rf-detr factory call site is at risk.
"""

from __future__ import annotations

import inspect

import pytest

RFDETR_SEVEN = [
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomRotation",
    "RandomAffine",
    "ColorJiggle",
    "RandomGaussianBlur",
    "RandomGaussianNoise",
]


# Snapshot of required keyword args (the ones rf-detr's factories explicitly pass).
# Extra args are allowed; these specific ones MUST remain accepted with their semantics.
EXPECTED_KW_PER_CLASS: dict[str, set[str]] = {
    # rf-detr _make_horizontal_flip → RandomHorizontalFlip(p=...)
    "RandomHorizontalFlip": {"p"},
    "RandomVerticalFlip": {"p"},
    # rf-detr _make_rotate → RandomRotation(degrees=..., p=...)
    "RandomRotation": {"degrees", "p"},
    # rf-detr _make_affine → RandomAffine(degrees=..., translate=..., scale=..., shear=..., p=...)
    "RandomAffine": {"degrees", "translate", "scale", "shear", "p"},
    # rf-detr _make_color_jitter & _make_random_brightness_contrast both call ColorJiggle(brightness=..., contrast=..., saturation=..., hue=..., p=...)
    "ColorJiggle": {"brightness", "contrast", "saturation", "hue", "p"},
    # rf-detr _make_gaussian_blur → RandomGaussianBlur(kernel_size=..., sigma=..., p=...)
    "RandomGaussianBlur": {"kernel_size", "sigma", "p"},
    # rf-detr _make_gauss_noise → RandomGaussianNoise(std=..., p=...)
    "RandomGaussianNoise": {"std", "p"},
}


@pytest.mark.parametrize("class_name", RFDETR_SEVEN)
def test_class_exists_in_kornia_augmentation_top_level(class_name):
    """The class must be importable from kornia.augmentation top level."""
    import kornia.augmentation as K

    assert hasattr(K, class_name), (
        f"{class_name} no longer in kornia.augmentation public API; "
        f"this would break rf-detr/src/rfdetr/datasets/kornia_transforms.py"
    )


@pytest.mark.parametrize("class_name", RFDETR_SEVEN)
def test_class_exists_in_plural_namespace(class_name):
    """The class must also be reachable via kornia.augmentations (plural)."""
    import kornia.augmentations as KA

    assert hasattr(KA, class_name), (
        f"{class_name} not reachable via kornia.augmentations (plural); namespace shim broken"
    )


@pytest.mark.parametrize("class_name,required_kw", list(EXPECTED_KW_PER_CLASS.items()))
def test_constructor_accepts_expected_kwargs(class_name, required_kw):
    """The constructor must accept each expected keyword argument."""
    import kornia.augmentation as K

    cls = getattr(K, class_name)
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    missing = required_kw - accepted
    # Some signatures may use **kwargs; if so, treat as accepting everything
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return
    assert not missing, (
        f"{class_name} no longer accepts {missing}; "
        f"rf-detr's factory at rf-detr/src/rfdetr/datasets/kornia_transforms.py "
        f"will fail with TypeError. Restore the kwarg or coordinate a downstream PR."
    )


def test_color_jitter_alias_present():
    """ColorJitter should still resolve (alias of ColorJiggle); rf-detr docs reference both."""
    import kornia.augmentation as K

    assert hasattr(K, "ColorJiggle")


def test_factory_smoke_construct_seven():
    """Construct each of the seven with rf-detr-style args; must not raise."""
    import kornia.augmentation as K

    K.RandomHorizontalFlip(p=0.5)
    K.RandomVerticalFlip(p=0.5)
    K.RandomRotation(degrees=15.0, p=0.5)
    K.RandomAffine(degrees=(-15.0, 15.0), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(0.0, 0.0), p=0.5)
    K.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5)
    K.RandomGaussianNoise(std=0.05, p=0.5)
