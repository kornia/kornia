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

"""Models submodule for Kornia.

This package provides model architectures and utilities for state-of-the-art models for Visual Language
Models and Vision Language Action Models.
"""

__all__ = [
    "depth_estimation",
    "detection",
    "edge_detection",
    "sam3",
    "segmentation",
    "super_resolution",
    "tracking",
]


def __getattr__(name: str):
    """Lazy load submodules to avoid circular imports."""
    if name in __all__:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
