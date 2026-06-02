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

import torch

import kornia.augmentation as K


def _build_rotation(cfg):
    r"""Build a RandomRotation augmentation module.

    Args:
        cfg: Configuration dictionary containing augmentation parameters.

    Returns:
        Initialized ``RandomRotation`` module.
    """
    return K.RandomRotation(degrees=cfg.get("degrees", 15.0), p=1.0)


def _build_blur(cfg):
    r"""Build a RandomGaussianBlur augmentation module.

    Args:
        cfg: Configuration dictionary containing augmentation parameters.

    Returns:
        Initialized ``RandomGaussianBlur`` module.
    """
    return K.RandomGaussianBlur(
        kernel_size=cfg.get("kernel_size", (3, 3)),
        sigma=cfg.get("sigma", (0.1, 2.0)),
        p=1.0,
    )


def _build_brightness(cfg):
    r"""Build a RandomBrightness augmentation module.

    Args:
        cfg: Configuration dictionary containing augmentation parameters.

    Returns:
        Initialized ``RandomBrightness`` module.
    """
    return K.RandomBrightness(
        brightness=cfg.get("brightness", (0.8, 1.2)),
        p=1.0,
    )


AUGMENTATION_REGISTRY = {
    "rotation": _build_rotation,
    "blur": _build_blur,
    "brightness": _build_brightness,
}


class AugmentationPipeline:
    def __init__(self, augmentations_config=None, mode="sequential"):
        r"""Initialize an augmentation pipeline.

        The pipeline can operate either by applying all augmentations
        sequentially or by applying each augmentation independently.

        Args:
            augmentations_config: List of augmentation configurations, where each
                configuration is a dictionary with keys:
                - ``"name"``: Name of the augmentation.
                - ``"params"``: Optional parameters for the augmentation.
            mode: Execution mode of the pipeline:
                - ``"sequential"``: Apply all augmentations in sequence.
                - ``"individual"``: Apply each augmentation independently.

        Raises:
            ValueError: If an unknown augmentation name is provided.
        """
        self.mode = mode
        self.augmentations = []

        if augmentations_config is None:
            augmentations_config = [
                {"name": "rotation", "params": {}},
                {"name": "blur", "params": {}},
                {"name": "brightness", "params": {}},
            ]

        for aug in augmentations_config:
            name = aug["name"]
            params = aug.get("params", {})

            if name not in AUGMENTATION_REGISTRY:
                raise ValueError(f"Unknown augmentation: {name}")

            module = AUGMENTATION_REGISTRY[name](params)
            self.augmentations.append((name, module))

        if self.mode == "sequential":
            modules = [module for _, module in self.augmentations]
            self.pipeline = torch.nn.Sequential(*modules)

    def __call__(self, x):
        r"""Apply augmentations to the input tensor.

        Depending on the selected mode, either applies all augmentations
        sequentially or applies each augmentation independently.

        Args:
            x: Input tensor to be augmented.

        Returns:
            Dictionary mapping augmentation names to augmented outputs:
            - In ``"sequential"`` mode, a single entry with the combined name.
            - In ``"individual"`` mode, one entry per augmentation.

        Raises:
            ValueError: If an unknown mode is specified.
        """
        if self.mode == "sequential":
            combined_name = "+".join([name for name, _ in self.augmentations])
            return {combined_name: self.pipeline(x)}

        elif self.mode == "individual":
            outputs = {}
            for name, aug in self.augmentations:
                outputs[name] = aug(x)
            return outputs

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
