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

import warnings
from collections import defaultdict

import torch
from feature_extractor import FeatureExtractor

# Lazy pandas import (Kornia style)
from kornia.core.external import LazyLoader

# Metric registry
from kornia.profiler.metrics.cosine import cosine_similarity
from kornia.profiler.metrics.gram import gram_similarity
from kornia.profiler.metrics.l2 import l2_normalized
from kornia.profiler.metrics.linear import linear_similarity

pandas = LazyLoader("pandas")

METRIC_REGISTRY = {
    "cosine": cosine_similarity,
    "linear": linear_similarity,
    "gram": gram_similarity,
    "l2": l2_normalized,
}


class ModelProfiler:
    def __init__(self, model, layers=None, processing="flatten"):
        r"""Initialize the model profiler.

        This class captures intermediate representations from selected layers
        of a model and computes similarity metrics between different input groups.

        Args:
            model: PyTorch model to be profiled.
            layers: List of layer names to extract features from. If ``None``,
                all layers are used.
            processing: Feature processing method applied to extracted outputs.
                Common options include ``"flatten"`` and ``"none"``.
        """
        self.model = model
        self.model.eval()

        self.layers = layers
        self.processing = processing

        self.storage = defaultdict(list)

        self.results = None
        self._df = None

        self.extractor = None

    def __enter__(self):
        r"""Enter context manager and initialize feature extraction hooks.

        Returns:
            The current ``ModelProfiler`` instance.
        """
        self.extractor = FeatureExtractor(self.model, self.layers, processing=self.processing)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        r"""Exit context manager and remove registered hooks."""
        if hasattr(self.extractor, "remove_hooks"):
            self.extractor.remove_hooks()

    def __call__(self, group="default", tag=None, **inputs):
        r"""Run a forward pass and store extracted features.

        Features are grouped and optionally tagged for later comparison.

        Args:
            group: Name of the group to which the input belongs.
            tag: Optional identifier for the input (e.g., augmentation name).
            **inputs: Keyword arguments passed to the model's forward method.
        """
        with torch.no_grad():
            features = self.extractor(**inputs)

        self.storage[group].append({"features": features, "tag": tag})

    def compute(self, metrics, groups):
        r"""Compute similarity metrics between two groups of inputs.

        The comparison is performed pairwise between corresponding entries
        in the specified groups.

        Args:
            metrics: List of metric names to compute.
            groups: List of exactly two group names to compare.

        Raises:
            ValueError: If the number of groups is not two or if an unknown
                metric is specified.
            RuntimeError: If one of the groups has no stored data.
        """
        if len(groups) != 2:
            raise ValueError("Currently supports pairwise group comparison")
        g1, g2 = groups
        feats1 = self.storage[g1]
        feats2 = self.storage[g2]

        if len(feats1) == 0 or len(feats2) == 0:
            raise RuntimeError("One of the groups has no data.")

        if len(feats1) != len(feats2):
            warnings.warn("Unequal group sizes, truncating to minimum length.", stacklevel=2)

        min_len = min(len(feats1), len(feats2))

        results = []

        for i in range(min_len):
            f1_entry = feats1[i]
            f2_entry = feats2[i]

            f1 = f1_entry["features"]
            f2 = f2_entry["features"]

            tag = f2_entry.get("tag", None) or f1_entry.get("tag", None)

            row = {}

            if tag is not None:
                row["tag"] = tag

            for metric_name in metrics:
                if metric_name not in METRIC_REGISTRY:
                    raise ValueError(f"Unknown metric: {metric_name}")

                metric_fn = METRIC_REGISTRY[metric_name]
                metric_output = metric_fn(f1, f2)

                for layer, val in metric_output.items():
                    if val is None:
                        continue
                    key = f"{metric_name}_{layer}"
                    row[key] = val

            results.append(row)

        self.results = results
        self._df = pandas.DataFrame(results)

    def print(self):
        r"""Print the computed results as a DataFrame."""
        if self._df is not None:
            print(self._df)
        else:
            print("No results computed yet.")

    def save_as_report(self, path):
        r"""Save computed results to a file.

        Supported formats include CSV and JSON.

        Args:
            path: File path where the report will be saved.

        Raises:
            RuntimeError: If results have not been computed.
            ValueError: If the file format is unsupported.
        """
        if self._df is None:
            raise RuntimeError("Run compute() before saving report.")

        if path.endswith(".csv"):
            self._df.to_csv(path, index=False)
        elif path.endswith(".json"):
            self._df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError("Unsupported format. Use .csv or .json")

    @property
    def df(self):
        r"""Return the results as a pandas DataFrame.

        Returns:
            DataFrame containing computed similarity metrics, or ``None``
            if results have not been computed yet.
        """
        return self._df
