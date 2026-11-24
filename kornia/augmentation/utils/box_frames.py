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

"""Describe the coordinate frame of every box and keypoint data key (#6710).

Two frames name the same pixel grid. In the **edge** frame of COCO and torchvision, pixel ``i`` spans ``[i, i + 1)``.
In the **pixel-centre** frame of kornia's image matrices and of :class:`~kornia.geometry.keypoints.Keypoints`, pixel
``i`` is centred at ``i`` and spans ``[i - 1/2, i + 1/2)``. A coordinate ``x`` of the pixel-centre frame is ``x + 1/2``
in the edge frame.
"""

import warnings
from typing import Iterable, NamedTuple, Set

from kornia.constants import DataKey
from kornia.core.utils import is_compiling


class BoxFrameWarning(UserWarning):
    """Warn that an augmentation container reads boxes or keypoints in a frame the data key does not name.

    It is raised once per process for each bare ``"bbox"`` and ``"bbox_xyxy"`` key, and once when one call mixes the
    edge and pixel-centre frames. A library that has checked its frames can silence it with
    ``warnings.filterwarnings("ignore", category=BoxFrameWarning)``.

    See Also:
        - :class:`~kornia.constants.DataKey`: the frame-named keys.
    """


class _BoxKey(NamedTuple):
    """How :class:`~kornia.augmentation.container.AugmentationSequential` reads and writes one box key.

    ``pixel_centre`` selects the pixel-centre storage (``Boxes._inclusive``), which applies image matrices to the
    corners as they are, over the exclusive storage, which applies them in the edge frame. A ``bare`` key imports a
    tensor as it did before #6710, in ``mode_out`` with the legacy validation, and reads an all-zero row as a real box at
    the origin. Every other key imports in the exclusive ``mode_in``, rejects negative extents and reads a zero width or
    height as a missing annotation. Both write their result in ``mode_out``.
    """

    mode_in: str
    mode_out: str
    pixel_centre: bool
    bare: bool


_BOX_KEYS = {
    # The bare keys are aliases of the pixel-centre keys: the same stored corners, matrices and region rules.
    DataKey.BBOX: _BoxKey("vertices_plus", "vertices_plus", pixel_centre=True, bare=True),
    DataKey.BBOX_XYXY: _BoxKey("xyxy_plus", "xyxy_plus", pixel_centre=True, bare=True),
    DataKey.BBOX_INDEX: _BoxKey("vertices", "vertices_plus", pixel_centre=True, bare=False),
    DataKey.BBOX_XYXY_INDEX: _BoxKey("xyxy", "xyxy_plus", pixel_centre=True, bare=False),
    DataKey.BBOX_EDGE: _BoxKey("vertices", "vertices", pixel_centre=False, bare=False),
    DataKey.BBOX_XYXY_EDGE: _BoxKey("xyxy", "xyxy", pixel_centre=False, bare=False),
    DataKey.BBOX_XYWH: _BoxKey("xywh", "xywh", pixel_centre=False, bare=False),
}
_KEYPOINT_KEYS = (DataKey.KEYPOINTS, DataKey.KEYPOINTS_EDGE)
_EDGE_KEYS = frozenset({DataKey.BBOX_EDGE, DataKey.BBOX_XYXY_EDGE, DataKey.BBOX_XYWH, DataKey.KEYPOINTS_EDGE})
_BARE_KEY_REPLACEMENTS = {
    DataKey.BBOX: ("bbox_index", "bbox_edge"),
    DataKey.BBOX_XYXY: ("bbox_xyxy_index", "bbox_xyxy_edge"),
}
_warned: Set[str] = set()


def _box_key(key: DataKey) -> _BoxKey:
    """Return how a box ``key`` is read, or raise ``ValueError`` for a key that holds no boxes."""
    if key not in _BOX_KEYS:
        raise ValueError(f"Unsupported mode `{key.name}`.")
    return _BOX_KEYS[key]


def _warn_once(tag: str, message: str) -> None:
    if tag in _warned:
        return
    _warned.add(tag)
    warnings.warn(message, BoxFrameWarning, stacklevel=4)


def _warn_about_frames(keys: Iterable[DataKey]) -> None:
    """Warn once per process about a bare box key, and about a call that reads two frames.

    Graph capture skips the check: the eager calls before it, or the construction of the container, already warned.
    """
    if is_compiling():
        return
    keys = list(keys)
    for key in keys:
        if key in _BARE_KEY_REPLACEMENTS:
            index_key, edge_key = _BARE_KEY_REPLACEMENTS[key]
            bare = key.name.lower()
            _warn_once(
                bare,
                f'"{bare}" reads boxes as corners in the pixel-centre frame, where pixel i is centred at i, as'
                f' keypoints are. Name the frame: "{index_key}" keeps these numbers, and "{edge_key}" reads boxes in'
                f" the edge frame of COCO and torchvision, where pixel i spans [i, i + 1). A -0.5 / +0.5 conversion"
                f' around this key keeps working with "{index_key}"; without it, use "{edge_key}". See kornia#6710.',
            )
    box_frames = {key in _EDGE_KEYS for key in keys if key in _BOX_KEYS}
    point_frames = {key in _EDGE_KEYS for key in keys if key in _KEYPOINT_KEYS}
    if box_frames and point_frames and box_frames != point_frames:
        _warn_once(
            "mixed",
            "This call reads boxes and keypoints in different frames: the edge frame (pixel i spans [i, i + 1)) for"
            ' "bbox_xywh", "bbox_edge", "bbox_xyxy_edge" and "keypoints_edge", and the pixel-centre frame (pixel i'
            ' centred at i) for the other box keys and "keypoints". One annotation file is in one frame: pick the'
            " keys of that frame. See kornia#6710.",
        )
