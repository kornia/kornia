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

import logging
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, DefaultDict, Dict, List, Optional, Union

import torch
from torch import Tensor, nn

from kornia.core.external import LazyLoader
from kornia.profiler.collectors.base import Collector
from kornia.profiler.metrics import (
    CosineSimilarity,
    GramSimilarity,
    L2Normalized,
    LinearSimilarity,
    Metric,
)
from kornia.profiler.transforms import Transform

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

pandas = LazyLoader("pandas")

MetricType = Union[Metric, str]
TransformType = Union[nn.Module, Callable[[Tensor], Tensor], Transform, str, None]

METRIC_REGISTRY: Dict[str, Callable[[], Metric]] = {
    "cosine": CosineSimilarity,
    "gram": GramSimilarity,
    "l2": L2Normalized,
    "linear": LinearSimilarity,
}


class Profiler:
    """Profile and compare model representations using configurable metrics.

    The profiler collects intermediate representations from a model and
    compares representations recorded under different input groups.

    Metrics can be specified by their registered names or as instances of
    ``Metric`` subclasses. An optional transform can be applied to each
    representation before metric evaluation.

    Args:
        model: PyTorch model whose representations will be collected.
            The model is switched to evaluation mode during initialization.
        collector: Collector responsible for attaching hooks and extracting
            representations from the model.
        pre_metric_transform: Optional transformation applied to each
            representation before metric evaluation. Supported values are
            a callable, a ``Transform`` instance, ``"flatten"``, or ``None``.
        metrics: List of registered metric names or ``Metric`` instances.
            Defaults to ``["cosine", "linear"]`` when ``None``.

    Raises:
        TypeError: If a metric or transform has an unsupported type.
        ValueError: If a metric name or transform name is unknown.
    """

    def __init__(
        self,
        model: nn.Module,
        collector: Collector,
        pre_metric_transform: TransformType = None,
        metrics: Optional[List[MetricType]] = None,
    ) -> None:
        self.model: nn.Module = model
        self.model.eval()
        self.collector: Collector = collector
        self.pre_metric_transform: Optional[Callable[[Tensor], Tensor]] = self._build_transform(pre_metric_transform)

        # Validate metrics before attaching hooks to the model.
        self.metrics: Dict[str, Metric] = self._build_metrics(metrics if metrics is not None else ["cosine", "linear"])

        try:
            self.collector.attach(model)
        except Exception:
            self.collector.remove_hooks()
            raise

        self.storage: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.results: Optional[List[Dict[str, Any]]] = None
        self._df: Optional[pd.DataFrame] = None

    @staticmethod
    def _build_transform(
        transform: TransformType,
    ) -> Optional[Callable[[Tensor], Tensor]]:
        """Build the optional transformation applied before metric evaluation.

        Args:
            transform: Callable, ``Transform`` instance, ``"flatten"``,
                or ``None``.

        Returns:
            A callable that transforms a tensor, or ``None`` when no
            transformation is configured.

        Raises:
            TypeError: If the transform is not supported.
            ValueError: If the transform name is unknown.
        """
        if transform is None:
            return None

        if isinstance(transform, Transform):
            return transform

        if isinstance(transform, str):
            if transform == "flatten":
                return lambda x: torch.flatten(x, start_dim=1)
            raise ValueError(f"Unknown pre-metric transform: {transform}")

        if callable(transform):
            return transform

        raise TypeError("pre_metric_transform must be callable, 'flatten', or None.")

    @staticmethod
    def _build_metrics(metrics: List[MetricType]) -> Dict[str, Metric]:
        """Resolve registered metric names and validate metric instances.

        Args:
            metrics: List containing registered metric names or ``Metric``
                instances.

        Returns:
            A dictionary mapping metric names to instantiated metric modules.

        Raises:
            TypeError: If an entry is neither a string nor a ``Metric``
                instance.
            ValueError: If a string does not identify a registered metric.
        """
        built_metrics: Dict[str, Metric] = {}

        for metric in metrics:
            if isinstance(metric, str):
                if metric not in METRIC_REGISTRY:
                    raise ValueError(f"Unknown metric: {metric}")
                built_metrics[metric] = METRIC_REGISTRY[metric]()
            elif isinstance(metric, Metric):
                built_metrics[metric.__class__.__name__] = metric
            else:
                raise TypeError("Each metric must be a Metric instance or a registered name.")

        return built_metrics

    def __call__(
        self,
        group: str = "default",
        tag: Optional[str] = None,
        **inputs: Tensor,
    ) -> None:
        """Collect and store representations for a named input group.

        The collector executes the model and returns the representations
        captured by its hooks. The representations and optional tag are
        appended to the storage associated with ``group``.

        Args:
            group: Name identifying the input group.
            tag: Optional label associated with this collection.
            **inputs: Tensor-valued keyword arguments passed to the collector
                and subsequently to the model.
        """
        with torch.no_grad():
            features = self.collector(**inputs)

        self.storage[group].append({"features": features, "tag": tag})

    def compute(self, groups: List[str]) -> None:
        """Compare stored representations from exactly two named groups.

        Corresponding entries are compared in insertion order. If the groups
        contain different numbers of entries, the comparison is truncated to
        the smaller group. Each configured metric is evaluated for every
        matching representation key.

        The results are stored in ``results`` and converted to a pandas
        DataFrame when computation completes. If a ``GramSimilarity``
        evaluation raises ``ValueError``, that metric result is skipped for
        the affected representation.

        Args:
            groups: Exactly two group names identifying the representations
                to compare.

        Raises:
            ValueError: If the number of groups is not two, or if the
                compared entries have different representation keys.
            RuntimeError: If either group contains no collected entries.
        """
        if len(groups) != 2:
            raise ValueError("Currently supports pairwise group comparison.")

        group_a, group_b = groups
        entries_a = self.storage[group_a]
        entries_b = self.storage[group_b]

        if not entries_a or not entries_b:
            raise RuntimeError("One of the groups has no data.")

        if len(entries_a) != len(entries_b):
            warnings.warn(
                "Unequal group sizes, truncating to minimum length.",
                stacklevel=2,
            )

        results: List[Dict[str, Any]] = []

        for entry_a, entry_b in zip(entries_a, entries_b):
            features_a = entry_a["features"]
            features_b = entry_b["features"]
            tag = entry_b.get("tag") or entry_a.get("tag")
            row: Dict[str, Any] = {}

            if tag is not None:
                row["tag"] = tag

            if features_a.keys() != features_b.keys():
                raise ValueError("Collected representations must have matching keys.")

            for layer in features_a:
                x = features_a[layer]
                y = features_b[layer]

                if self.pre_metric_transform is not None:
                    x = self.pre_metric_transform(x)
                    y = self.pre_metric_transform(y)

                for metric_name, metric in self.metrics.items():
                    try:
                        score = metric(x, y)
                    except ValueError:
                        if isinstance(metric, GramSimilarity):
                            continue
                        raise

                    row[f"{metric_name}_{layer}"] = score.item()

            results.append(row)

        self.results = results
        self._df = pandas.DataFrame(results)

    def print(self) -> None:
        """Print the computed results, or indicate that none are available."""
        if self._df is not None:
            print(self._df)
        else:
            print("No results computed yet.")

    def save_as_report(self, path: str) -> None:
        """Save computed results as a CSV or JSON report.

        The output format is selected from the file extension. CSV files are
        written without an index, while JSON files use a record-oriented
        structure.

        Args:
            path: Destination file path ending in ``.csv`` or ``.json``.

        Raises:
            RuntimeError: If ``compute`` has not produced a DataFrame.
            ValueError: If the file extension is unsupported.
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
    def df(self) -> Optional["pd.DataFrame"]:
        """Return the computed results DataFrame, or ``None`` if not computed."""
        return self._df
