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

"""A pickled object with an observable side effect, for tests of ``weights_only`` loading.

Store a :class:`CreatesMarkerOnLoad` in a checkpoint and load it through the code under
test with :func:`load_without_running_payload`. The only effect of unpickling the object is
an empty marker file, so the marker existing afterwards means the loader executed a callable
named in the file it was handed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

__all__ = ["CreatesMarkerOnLoad", "create_marker", "load_without_running_payload"]


def create_marker(path: str) -> None:
    """Create an empty file at *path*; the only effect of unpickling :class:`CreatesMarkerOnLoad`."""
    Path(path).touch()


class CreatesMarkerOnLoad:
    """An object whose unpickling calls :func:`create_marker`, standing in for any pickled callable."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (create_marker, (str(self.marker),))


def load_without_running_payload(marker: Path, load: Callable[[], object]) -> object:
    """Return ``load()``, failing if *marker* exists afterwards, whether ``load`` returned or raised.

    Checking in ``finally`` puts the payload having run ahead of whatever the loader then
    returns or raises, so a loader that executes it fails on exactly that.
    """
    try:
        return load()
    finally:
        assert not marker.exists(), "loading the checkpoint ran a callable pickled into it"
