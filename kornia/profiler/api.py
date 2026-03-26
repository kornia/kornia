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
    """Load config from dict or JSON file."""
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
    """Generic representation comparison between two inputs."""
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
    """Wrapper over generic API for augmentation-based evaluation."""
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
