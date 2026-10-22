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

import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

import kornia.augmentation as K

# Compatibility aliases for the names accepted by the previous configuration API.
# These map names to augmentation classes; they do not construct augmentations.
_LEGACY_AUGMENTATION_NAMES = {
    "rotation": "RandomRotation",
    "blur": "RandomGaussianBlur",
    "brightness": "RandomBrightness",
}


def _build_augmentation(config: Dict[str, Any]) -> Tuple[str, torch.nn.Module]:
    """Resolve and construct a Kornia augmentation from its name and arguments.

    The preferred configuration format places constructor arguments alongside
    ``name``::

        {"name": "RandomRotation", "degrees": 30, "p": 1.0}

    The previous ``{"name": "rotation", "params": {...}}`` format remains
    supported for compatibility.

    Args:
        config: Dictionary containing the augmentation name and its constructor
            arguments.

    Returns:
        A tuple containing the configured augmentation name and the
        instantiated augmentation module.

    Raises:
        TypeError: If the configuration, its name, its ``params`` field, or
            the augmentation constructor arguments have an invalid type or
            cannot be bound to the constructor signature.
        ValueError: If the configuration is missing ``name``, the augmentation
            name is unknown, or the legacy ``params`` field is mixed with
            top-level constructor arguments.
    """
    if not isinstance(config, dict):
        raise TypeError("Each augmentation configuration must be a dictionary.")

    if "name" not in config:
        raise ValueError("Each augmentation configuration must include 'name'.")

    configured_name = config["name"]
    if not isinstance(configured_name, str) or not configured_name:
        raise TypeError("Augmentation 'name' must be a non-empty string.")

    name = _LEGACY_AUGMENTATION_NAMES.get(configured_name, configured_name)
    augmentation_cls = getattr(K, name, None)

    if augmentation_cls is None or not inspect.isclass(augmentation_cls):
        raise ValueError(f"Unknown Kornia augmentation: {configured_name}")
    if not issubclass(augmentation_cls, torch.nn.Module):
        raise TypeError(f"Kornia augmentation {name!r} is not a torch.nn.Module class.")

    if "params" in config:
        params = config["params"]
        if not isinstance(params, dict):
            raise TypeError("Augmentation 'params' must be a dictionary.")

        extra_params = set(config) - {"name", "params"}
        if extra_params:
            raise ValueError(
                f"Do not mix the legacy 'params' field with top-level constructor arguments: {sorted(extra_params)}"
            )

        params = dict(params)
    else:
        params = {key: value for key, value in config.items() if key != "name"}

    # The old builders always applied their augmentation (p=1.0). Preserve
    # that behavior for legacy aliases while allowing the new class-name
    # configuration format to use the constructor's own defaults.
    if configured_name in _LEGACY_AUGMENTATION_NAMES:
        params.setdefault("p", 1.0)

    # Keep the previous brightness configuration key working.
    if configured_name == "brightness" and "brightness_factor" in params:
        if "brightness" in params:
            raise ValueError("Specify only one of 'brightness_factor' and 'brightness'.")
        params["brightness"] = params.pop("brightness_factor")

    try:
        signature = inspect.signature(augmentation_cls)
        signature.bind(**params)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Invalid arguments for Kornia augmentation {name!r}: {exc}") from exc

    module = augmentation_cls(**params)
    return configured_name, module


class AugmentationPipeline:
    """Configure and apply Kornia augmentations to input tensors.

    Augmentations can be applied sequentially as a combined pipeline or
    independently to the same input tensor.

    The preferred configuration format uses the augmentation class name and
    constructor arguments directly, for example
    ``{"name": "RandomRotation", "degrees": 30, "p": 1.0}``.

    The legacy ``name`` plus nested ``params`` format and the aliases
    ``rotation``, ``blur``, and ``brightness`` remain supported.

    Attributes:
        mode: Execution mode, either ``"sequential"`` or ``"individual"``.
        augmentations: List of configured augmentation names and their
            instantiated modules.
        pipeline: Sequential container of augmentation modules. Available
            when ``mode`` is ``"sequential"``.
    """

    def __init__(
        self,
        augmentations_config: Optional[List[Dict[str, Any]]] = None,
        mode: str = "sequential",
    ) -> None:
        r"""Initialize an augmentation pipeline.

        Each configuration identifies a Kornia augmentation class by name and
        supplies constructor arguments directly, for example
        ``{"name": "RandomRotation", "degrees": 30, "p": 1.0}``.

        The previous ``name`` plus nested ``params`` format and the names
        ``rotation``, ``blur``, and ``brightness`` remain supported for
        compatibility.

        Args:
            augmentations_config: List of augmentation configuration
                dictionaries. If ``None``, a default rotation, blur, and
                brightness pipeline is created.
            mode: ``"sequential"`` applies all augmentations in sequence;
                ``"individual"`` applies each augmentation independently
                to the original input.

        Raises:
            TypeError: If a configuration or constructor argument has an
                invalid type, or if constructor arguments cannot be bound
                to the augmentation constructor.
            ValueError: If an augmentation name or execution mode is unknown,
                if incompatible configuration formats are combined, or if
                individual mode contains duplicate augmentation names.
        """
        if mode not in {"sequential", "individual"}:
            raise ValueError(f"Unknown mode: {mode}")

        self.mode: str = mode
        self.augmentations: List[Tuple[str, torch.nn.Module]] = []

        if augmentations_config is None:
            augmentations_config = [
                {"name": "RandomRotation", "degrees": 15.0, "p": 1.0},
                {
                    "name": "RandomGaussianBlur",
                    "kernel_size": (3, 3),
                    "sigma": (0.1, 2.0),
                    "p": 1.0,
                },
                {
                    "name": "RandomBrightness",
                    "brightness": (0.8, 1.2),
                    "p": 1.0,
                },
            ]

        seen_names: set[str] = set()
        for config in augmentations_config:
            name, augmentation = _build_augmentation(config)

            if self.mode == "individual" and name in seen_names:
                raise ValueError("Augmentation names must be unique in individual mode.")

            seen_names.add(name)
            self.augmentations.append((name, augmentation))

        if self.mode == "sequential":
            modules = [module for _, module in self.augmentations]
            self.pipeline: torch.nn.Sequential = torch.nn.Sequential(*modules)

    def __call__(self, x: Tensor) -> Dict[str, Tensor]:
        r"""Apply the configured augmentations to an input tensor.

        In sequential mode, all augmentations are applied in order and the
        result is returned under a key formed by joining their configured
        names with ``"+"``. In individual mode, each augmentation is applied
        independently to the original input, and each result is returned
        under its configured name.

        Args:
            x: Input tensor accepted by the configured Kornia augmentations.

        Returns:
            A dictionary mapping augmentation names to their augmented
            tensors. Sequential mode returns one entry for the combined
            pipeline; individual mode returns one entry per augmentation.
        """
        if self.mode == "sequential":
            combined_name = "+".join(name for name, _ in self.augmentations)
            return {combined_name: self.pipeline(x)}

        outputs: Dict[str, Tensor] = {}
        for name, augmentation in self.augmentations:
            outputs[name] = augmentation(x)
        return outputs