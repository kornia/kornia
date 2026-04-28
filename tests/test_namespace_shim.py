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

"""Tests for the kornia.augmentations (plural) namespace shim."""

from __future__ import annotations

import importlib
import sys

import pytest


class TestTopLevelReExport:
    """Top-level symbols from singular path are accessible via plural."""

    @pytest.mark.parametrize(
        "class_name",
        [
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "RandomAffine",
            "AugmentationSequential",
            "Normalize",
        ],
    )
    def test_class_accessible(self, class_name: str) -> None:
        import kornia.augmentations as plural

        assert hasattr(plural, class_name), f"{class_name} not found in kornia.augmentations"
        cls = getattr(plural, class_name)
        assert cls is not None

    def test_normalize_is_callable(self) -> None:
        from kornia.augmentations import Normalize

        assert callable(Normalize)

    def test_augmentation_sequential_is_callable(self) -> None:
        from kornia.augmentations import AugmentationSequential

        assert callable(AugmentationSequential)


class TestSubModuleAccess:
    """Sub-module access via plural path resolves correctly."""

    def test_2d_submodule_resolves(self) -> None:
        import kornia.augmentations

        mod = kornia.augmentations._2d
        assert mod is not None

    def test_3d_submodule_resolves(self) -> None:
        import kornia.augmentations

        mod = kornia.augmentations._3d
        assert mod is not None

    def test_container_submodule_resolves(self) -> None:
        import kornia.augmentations

        mod = kornia.augmentations.container
        assert mod is not None

    def test_2d_registered_in_sys_modules(self) -> None:
        import kornia.augmentations  # noqa: F401

        assert "kornia.augmentations._2d" in sys.modules

    def test_container_registered_in_sys_modules(self) -> None:
        import kornia.augmentations  # noqa: F401

        assert "kornia.augmentations.container" in sys.modules


class TestDeepPathImport:
    """Deep-path imports through the plural namespace resolve correctly."""

    def test_deep_import_horizontal_flip(self) -> None:
        from kornia.augmentations._2d.geometric.horizontal_flip import RandomHorizontalFlip

        assert RandomHorizontalFlip is not None
        assert callable(RandomHorizontalFlip)

    def test_deep_import_via_importlib(self) -> None:
        mod = importlib.import_module("kornia.augmentations._2d.geometric.horizontal_flip")
        assert hasattr(mod, "RandomHorizontalFlip")


class TestIdentityAssertion:
    """Singular and plural paths resolve to the same class objects."""

    @pytest.mark.parametrize(
        "class_name",
        [
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "RandomAffine",
            "AugmentationSequential",
            "Normalize",
        ],
    )
    def test_same_object(self, class_name: str) -> None:
        import kornia.augmentation as singular
        import kornia.augmentations as plural

        cls_singular = getattr(singular, class_name)
        cls_plural = getattr(plural, class_name)
        assert cls_singular is cls_plural, (
            f"{class_name}: singular ({id(cls_singular)}) is not plural ({id(cls_plural)})"
        )

    def test_deep_path_identity(self) -> None:
        from kornia.augmentation._2d.geometric.horizontal_flip import (
            RandomHorizontalFlip as singular_cls,
        )
        from kornia.augmentations._2d.geometric.horizontal_flip import (
            RandomHorizontalFlip as plural_cls,
        )

        assert singular_cls is plural_cls


class TestAllSymbolsAccessible:
    """All __all__ symbols from singular are accessible via plural."""

    def test_all_exported(self) -> None:
        import kornia.augmentation as singular
        import kornia.augmentations as plural

        missing = []
        for name in singular.__all__:
            if not hasattr(plural, name):
                missing.append(name)

        assert not missing, f"Missing from kornia.augmentations: {missing}"

    def test_all_identical_objects(self) -> None:
        import kornia.augmentation as singular
        import kornia.augmentations as plural

        not_identical = []
        for name in singular.__all__:
            s = getattr(singular, name, None)
            p = getattr(plural, name, None)
            if s is not p:
                not_identical.append(name)

        assert not not_identical, f"Not identical objects: {not_identical}"
