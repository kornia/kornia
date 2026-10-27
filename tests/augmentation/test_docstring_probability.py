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

"""The first docstring line of an augmentation that is not always applied states its default ``p``.

``help()``, ``inspect.getdoc`` and API listings show the first docstring line first, and a reader who stops
there must not miss that ``RandomRotation(degrees=30.0)`` leaves about half of the batch unrotated. Every
public class in ``kornia.augmentation.__all__`` whose constructor defaults ``p`` below ``1`` therefore says
"probability ``p`` (default <value>)" on that line. A new class with such a default is picked up here.
"""

import inspect

import pytest

import kornia.augmentation as K


def _classes_with_partial_default_p() -> list[tuple[str, float]]:
    found = []
    for name in K.__all__:
        obj = getattr(K, name)
        if not inspect.isclass(obj) or "Base" in name:
            continue
        param = inspect.signature(obj.__init__).parameters.get("p")
        if param is None or not isinstance(param.default, (int, float)):
            continue
        if param.default < 1:
            found.append((name, param.default))
    return found


_CLASSES = _classes_with_partial_default_p()


def test_the_discovery_finds_the_known_classes():
    found = dict(_CLASSES)
    # A guard against a vacuous pass: the discovery must keep seeing the classes it was written for.
    assert found["RandomRotation"] == 0.5
    assert found["RandomGrayscale"] == 0.1
    assert found["RandomMosaic"] == 0.7
    assert found["RandomAffine3D"] == 0.5
    assert len(found) >= 49


@pytest.mark.parametrize(("name", "default"), _CLASSES, ids=[name for name, _ in _CLASSES])
def test_first_docstring_line_states_the_default_probability(name, default):
    first_line = (inspect.getdoc(getattr(K, name)) or "").splitlines()[0]

    assert f"probability ``p`` (default {default})" in first_line, (
        f"{name} defaults to p={default}; its first docstring line must say "
        f"'probability ``p`` (default {default})'. Got: {first_line!r}"
    )
