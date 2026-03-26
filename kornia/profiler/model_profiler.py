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
        self.model = model
        self.model.eval()

        self.layers = layers
        self.processing = processing

        self.storage = defaultdict(list)

        self.results = None
        self._df = None

        self.extractor = None

    def __enter__(self):
        self.extractor = FeatureExtractor(self.model, self.layers, processing=self.processing)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.extractor, "remove_hooks"):
            self.extractor.remove_hooks()

    def __call__(self, group="default", tag=None, **inputs):
        with torch.no_grad():
            features = self.extractor(**inputs)

        self.storage[group].append({"features": features, "tag": tag})

    def compute(self, metrics, groups):
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
        if self._df is not None:
            print(self._df)
        else:
            print("No results computed yet.")

    def save_as_report(self, path):
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
        return self._df
