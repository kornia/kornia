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

AUGMENTATION_REGISTRY = {
    "rotation": lambda cfg: K.RandomRotation(degrees=cfg.get("degrees", 15.0), p=1.0),
    "blur": lambda cfg: K.RandomGaussianBlur(
        kernel_size=cfg.get("kernel_size", (3, 3)),
        sigma=cfg.get("sigma", (0.1, 2.0)),
        p=1.0,
    ),
    "brightness": lambda cfg: K.RandomBrightness(
        brightness=cfg.get("brightness", (0.8, 1.2)),
        p=1.0,
    ),
}


class AugmentationPipeline:
    def __init__(self, augmentations_config=None, mode="sequential"):
        """Initialize augmentation pipeline."""
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
