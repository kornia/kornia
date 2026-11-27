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

import json
from typing import Any, Dict, Optional, Union

from torch import Tensor, nn

from kornia.profiler.collectors import LayerCollector
from kornia.profiler.engines import AugmentationPipeline
from kornia.profiler.profiler import Profiler


def _load_config(config: Optional[Union[Dict[str, Any], str]]) -> Dict[str, Any]:
    r"""Load configuration from a dictionary, JSON file, or ``None``.

    Args:
        config: Configuration dictionary, path to a JSON file, or ``None``.

    Returns:
        The configuration dictionary. Returns an empty dictionary when
        ``config`` is ``None``.

    Raises:
        json.JSONDecodeError: If the specified JSON file contains invalid JSON.
        OSError: If the specified file cannot be opened or read.
    """
    if config is None:
        return {}

    if isinstance(config, str):
        with open(config) as f:
            return json.load(f)

    return config


def model_profile_under_input_changes(
    model: nn.Module,
    input_a: Tensor,
    input_b: Tensor,
    config: Optional[Union[Dict[str, Any], str]] = None,
    output: Optional[str] = None,
) -> Profiler:
    r"""Profile model representations under two different inputs.

    The model must accept its input as a keyword argument named ``x``.
    Representations collected from the two inputs are compared using the
    configured metrics.

    Args:
        model: PyTorch model to profile.
        input_a: First input tensor.
        input_b: Second input tensor.
        config: Optional configuration dictionary or JSON file path.
            Supported settings include ``layers``, ``processing``, and
            ``metrics``.
        output: Optional destination path for a CSV or JSON report.

    Returns:
        A ``Profiler`` instance containing the computed results.

    Raises:
        TypeError: If a configuration value has an unsupported type.
        ValueError: If a configured layer, metric, or processing mode is
            invalid, or if the collected representations cannot be compared.
        RuntimeError: If representation collection or comparison fails.
    """
    config = _load_config(config)

    layers = config.get("layers")
    processing = config.get("processing", "flatten")
    metrics = config.get("metrics", ["cosine", "linear"])

    collector = LayerCollector(layers=layers, processing=processing)

    try:
        profiler = Profiler(model, collector=collector, metrics=metrics)

        profiler(x=input_a, group="group_a", tag="input_a")
        profiler(x=input_b, group="group_b", tag="input_b")

        profiler.compute(groups=["group_a", "group_b"])

        if output:
            profiler.save_as_report(output)

        return profiler
    finally:
        collector.remove_hooks()


def model_profile_under_augmentation(
    model: nn.Module,
    config: Union[Dict[str, Any], str],
    output: Optional[str] = None,
) -> Profiler:
    r"""Profile model representations under input augmentations.

    Each augmented input is compared independently against the original
    input. The resulting report contains one row per augmentation.

    The model must accept its input as a keyword argument named ``x``.

    Args:
        model: PyTorch model to profile.
        config: Configuration dictionary or JSON file path containing the
            input tensor and optional layer, augmentation, processing, and
            metric settings. The dictionary must contain an ``input`` entry.
        output: Optional destination path for a CSV or JSON report.

    Returns:
        A ``Profiler`` instance containing the computed results.

    Raises:
        KeyError: If the configuration does not contain the required
            ``input`` entry.
        TypeError: If a configuration value has an unsupported type.
        ValueError: If an augmentation, metric, layer, processing mode, or
            execution mode is invalid, or if the collected representations
            cannot be compared.
        RuntimeError: If representation collection or comparison fails.
    """
    config = _load_config(config)

    layers = config.get("layers")
    augment_config = config.get("augmentations")
    mode = config.get("mode", "individual")
    processing = config.get("processing", "flatten")
    metrics = config.get("metrics", ["cosine", "linear"])

    input_a = config["input"]

    augmenter = AugmentationPipeline(augment_config, mode=mode)
    aug_outputs = augmenter(input_a)

    collector = LayerCollector(layers=layers, processing=processing)

    try:
        profiler = Profiler(model, collector=collector, metrics=metrics)

        for aug_name, input_b in aug_outputs.items():
            profiler(x=input_a, group="group_a", tag="original")
            profiler(x=input_b, group="group_b", tag=aug_name)

        profiler.compute(groups=["group_a", "group_b"])

        if output:
            profiler.save_as_report(output)

        return profiler
    finally:
        collector.remove_hooks()
