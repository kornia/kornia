import json

from kornia.profiler.model_profiler import ModelProfiler
from kornia.profiler.augment import AugmentationPipeline


# -------------------- CONFIG LOADER --------------------

def _load_config(config):
    """
    Supports:
    - dict (direct use)
    - str (path to JSON file)
    """
    if config is None:
        return {}

    if isinstance(config, str):
        with open(config, "r") as f:
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
    """
    Generic representation comparison between two inputs
    """

    config = _load_config(config)

    layers = config.get("layers", None)
    processing = config.get("processing", "flatten")
    metrics = config.get("metrics", ["cosine", "linear"])

    with ModelProfiler(model, layers=layers, processing=processing) as p:

        p(x=input_a, group="group_a", tag="input_a")
        p(x=input_b, group="group_b", tag="input_b")

        p.compute(
            metrics=metrics,
            groups=["group_a", "group_b"]
        )

        if output:
            p.save_as_report(output)

        return p


# -------------------- AUGMENTATION WRAPPER --------------------

def model_profile_under_augmentation(
    model,
    config,
    output=None,
):
    """
    Wrapper over generic API for augmentation-based evaluation
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

        p.compute(
            metrics=metrics,
            groups=["group_a", "group_b"]
        )

        if output:
            p.save_as_report(output)

        return p