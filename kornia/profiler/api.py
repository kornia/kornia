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

from kornia.profiler.augment import AugmentationPipeline
from kornia.profiler.model_profiler import ModelProfiler

# -------------------- CONFIG LOADER --------------------


def _load_config(config):
    r"""Load configuration from a dictionary or a JSON file.

    If the input is ``None``, an empty dictionary is returned. If the input
    is a string, it is interpreted as a file path to a JSON configuration file.

    Args:
        config: Configuration provided either as a dictionary or a path to a JSON file.

    Returns:
        Dictionary containing configuration parameters.
    """
    if config is None:
        return {}

    if isinstance(config, str):
        with open(config) as f:
            return json.load(f)

    return config


# -------------------- CORE GENERIC API --------------------


def model_profile_under_input_changes(
    model,
    input_a,
    input_b,
    config=None,
    output=None,
):
    r"""Profile model representations under two different inputs.

    This function compares intermediate representations of a model when
    evaluated on two inputs. The comparison is performed layer-wise using
    the specified metrics.

    Args:
        model: PyTorch model to be profiled.
        input_a: First input tensor.
        input_b: Second input tensor.
        config: Optional configuration (dict or JSON path) specifying:
            - ``layers``: Layers to capture.
            - ``processing``: Feature processing method (e.g., ``"flatten"``).
            - ``metrics``: List of similarity metrics.
        output: Optional file path to save the profiling report.

    Returns:
        ModelProfiler instance containing computed results.
    """
    config = _load_config(config)

    layers = config.get("layers", None)
    processing = config.get("processing", "flatten")
    metrics = config.get("metrics", ["cosine", "linear"])

    with ModelProfiler(model, layers=layers, processing=processing) as p:
        p(x=input_a, group="group_a", tag="input_a")
        p(x=input_b, group="group_b", tag="input_b")

        p.compute(metrics=metrics, groups=["group_a", "group_b"])

        if output:
            p.save_as_report(output)

        return p


# -------------------- AUGMENTATION WRAPPER --------------------


def model_profile_under_augmentation(
    model,
    config,
    output=None,
):
    r"""Profile model representations under input augmentations.

    This function applies a set of augmentations to an input and compares
    the resulting representations against the original input using the
    specified similarity metrics.

    Args:
        model: PyTorch model to be profiled.
        config: Configuration (dict or JSON path) containing:
            - ``input``: Base input tensor.
            - ``layers``: Layers to capture.
            - ``augmentations``: Augmentation configuration.
            - ``mode``: Augmentation mode (e.g., ``"individual"``).
            - ``processing``: Feature processing method.
            - ``metrics``: List of similarity metrics.
        output: Optional file path to save the profiling report.

    Returns:
        ModelProfiler instance containing computed results.
    """
    config = _load_config(config)

    layers = config.get("layers", None)
    augment_config = config.get("augmentations", None)
    mode = config.get("mode", "individual")
    processing = config.get("processing", "flatten")
    metrics = config.get("metrics", ["cosine", "linear"])

    input_a = config["input"]

    augmenter = AugmentationPipeline(augment_config, mode=mode)
    aug_outputs = augmenter(input_a)

    with ModelProfiler(model, layers=layers, processing=processing) as p:
        for aug_name, input_b in aug_outputs.items():
            p(x=input_a, group="group_a", tag="original")
            p(x=input_b, group="group_b", tag=aug_name)

        p.compute(metrics=metrics, groups=["group_a", "group_b"])

        if output:
            p.save_as_report(output)

        return p
