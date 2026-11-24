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

"""Frame-named box and keypoint keys of ``AugmentationSequential`` (kornia#6710).

Expected values are written by hand from the two frames: pixel ``i`` spans ``[i, i + 1)`` in the edge frame and is
centred at ``i`` in the pixel-centre frame. A flip of a ``W``-pixel image maps the edge-frame ``x`` to ``W - x`` and the
pixel-centre ``x`` to ``W - 1 - x``; ``Resize`` 2x with ``align_corners=False`` maps the edge-frame ``x`` to ``2 x`` and
the pixel-centre ``x`` to ``2 x + 1/2``.
"""

import warnings

import pytest
import torch

import kornia.augmentation as K
from kornia.augmentation import BoxFrameWarning
from kornia.augmentation.utils import box_frames
from kornia.constants import DataKey

from testing.base import BaseTester

_BOX = [[[10.0, 12.0, 30.0, 40.0]]]
_CORNERS = [[[[10.0, 12.0], [30.0, 12.0], [30.0, 40.0], [10.0, 40.0]]]]


def _flip_and_resize():
    return (
        ("hflip", lambda: K.RandomHorizontalFlip(p=1.0)),
        ("resize2x", lambda: K.Resize((128, 128), align_corners=False)),
    )


@pytest.fixture
def fresh_warnings(monkeypatch):
    # The warnings fire once per process; give each test its own record.
    monkeypatch.setattr(box_frames, "_warned", set())


class TestConventionFrameNamedKeys(BaseTester):
    @pytest.mark.parametrize(
        "key, expected",
        [
            # edge frame: flip W - x, resize 2 x
            ("bbox_xyxy_edge", {"hflip": [34.0, 12.0, 54.0, 40.0], "resize2x": [20.0, 24.0, 60.0, 80.0]}),
            # pixel-centre frame: flip W - 1 - x, resize 2 x + 1/2, the numbers of the bare key
            ("bbox_xyxy_index", {"hflip": [33.0, 12.0, 53.0, 40.0], "resize2x": [20.5, 24.5, 60.5, 80.5]}),
            ("bbox_xyxy", {"hflip": [33.0, 12.0, 53.0, 40.0], "resize2x": [20.5, 24.5, 60.5, 80.5]}),
        ],
    )
    def test_convention_xyxy_keys_move_boxes_in_their_frame_6710(self, key, expected, device, dtype):
        image = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        box = torch.tensor(_BOX, device=device, dtype=dtype)
        for name, make in _flip_and_resize():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BoxFrameWarning)
                seq = K.AugmentationSequential(make(), data_keys=["input", key])
            out = seq(image, box)[1]
            self.assert_close(out, torch.tensor([[expected[name]]], device=device, dtype=dtype))

    @pytest.mark.parametrize(
        "key, hflip_x",
        [("bbox_edge", (34.0, 54.0)), ("bbox_index", (33.0, 53.0)), ("bbox", (33.0, 53.0))],
    )
    def test_convention_corner_keys_move_boxes_in_their_frame_6710(self, key, hflip_x, device, dtype):
        image = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        corners = torch.tensor(_CORNERS, device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoxFrameWarning)
            seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", key])
        out = seq(image, corners)[1]
        x0, x1 = hflip_x
        expected = torch.tensor([[[[x0, 12.0], [x1, 12.0], [x1, 40.0], [x0, 40.0]]]], device=device, dtype=dtype)
        self.assert_close(out, expected)

    def test_convention_keypoints_edge_moves_points_in_the_edge_frame_6710(self, device, dtype):
        image = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        points = torch.tensor([[[10.0, 12.0], [30.0, 40.0]]], device=device, dtype=dtype)
        expected = {
            "hflip": {"keypoints_edge": [[54.0, 12.0], [34.0, 40.0]], "keypoints": [[53.0, 12.0], [33.0, 40.0]]},
            "resize2x": {"keypoints_edge": [[20.0, 24.0], [60.0, 80.0]], "keypoints": [[20.5, 24.5], [60.5, 80.5]]},
        }
        for name, make in _flip_and_resize():
            for key in ("keypoints_edge", "keypoints"):
                seq = K.AugmentationSequential(make(), data_keys=["input", key])
                out = seq(image, points)[1]
                assert out.dtype == dtype
                self.assert_close(out, torch.tensor([expected[name][key]], device=device, dtype=dtype))

    def test_convention_edge_keypoints_land_on_edge_box_corners_6710(self, device, dtype):
        # One annotation file in the edge frame: the box corners and the same points move together.
        image = torch.rand(2, 3, 40, 52, device=device, dtype=dtype)
        box = torch.tensor([[[3.0, 5.0, 17.0, 29.0]], [[20.0, 1.0, 41.0, 13.0]]], device=device, dtype=dtype)
        points = box.reshape(2, 2, 2)  # (x1, y1) and (x2, y2)
        aug = K.RandomAffine(degrees=0.0, translate=(0.1, 0.1), scale=(0.7, 1.4), p=1.0)
        seq = K.AugmentationSequential(aug, data_keys=["input", "bbox_xyxy_edge", "keypoints_edge"])
        out_box, out_points = seq(image, box, points)[1:]
        self.assert_close(out_points.reshape(2, 1, 4), out_box)

    @pytest.mark.parametrize(
        "aug",
        [
            K.RandomAffine(degrees=25.0, translate=(0.1, 0.2), scale=(0.6, 1.5), shear=8.0, p=1.0),
            K.RandomPerspective(0.4, p=1.0),
            K.RandomResizedCrop((30, 50), p=1.0),
            K.RandomRotation(40.0, p=1.0),
        ],
    )
    def test_convention_bare_key_is_an_alias_of_the_index_key_6710(self, aug, device, dtype):
        # The bare keys are aliases of the pixel-centre keys: the same matrices on the same corners. They differ by float
        # rounding only: the bare "bbox_xyxy" builds its far corners as x1 + (x2 - x1 + 1) - 1, as before #6710, so
        # that its own outputs keep their bits, and "bbox_xyxy_index" stores x2 as given.
        image = torch.rand(2, 3, 48, 64, device=device, dtype=dtype)
        boxes = torch.tensor(
            [[[3.3, 4.7, 20.9, 30.1], [40.0, 2.0, 64.0, 48.0]], [[0.0, 0.0, 1.0, 1.0], [11.5, 17.25, 12.75, 30.0]]],
            device=device,
            dtype=dtype,
        )
        corners = torch.stack(
            [boxes[..., [0, 1]], boxes[..., [2, 1]], boxes[..., [2, 3]], boxes[..., [0, 3]]], dim=-2
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoxFrameWarning)
            bare = K.AugmentationSequential(aug, data_keys=["input", "bbox_xyxy", "bbox"])
        torch.manual_seed(0)
        out_bare = bare(image, boxes, corners)
        index = K.AugmentationSequential(aug, data_keys=["input", "bbox_xyxy_index", "bbox_index"])
        out_index = index(image, boxes, corners, params=bare._params)
        self.assert_close(out_index[1], out_bare[1])
        self.assert_close(out_index[2], out_bare[2])

    def test_convention_zero_rows_are_no_box_only_under_frame_named_keys_6710(self, device, dtype):
        image = torch.rand(1, 3, 64, 64, device=device, dtype=dtype)
        boxes = torch.tensor([[[10.0, 12.0, 30.0, 40.0], [0.0, 0.0, 0.0, 0.0]]], device=device, dtype=dtype)
        expected_rows = {
            "bbox_xyxy": [63.0, 0.0, 63.0, 0.0],  # a box at the origin, as before #6710
            "bbox_xyxy_index": [0.0, 0.0, 0.0, 0.0],
            "bbox_xyxy_edge": [0.0, 0.0, 0.0, 0.0],
        }
        for key, row in expected_rows.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BoxFrameWarning)
                seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", key])
            out = seq(image, boxes)[1]
            self.assert_close(out[0, 1], torch.tensor(row, device=device, dtype=dtype))

    def test_convention_index_key_rejects_inverted_boxes_the_bare_key_accepted_6710(self, device, dtype):
        # The bare key keeps its earlier check, an inclusive extent above zero, so x2 = x1 - 0.5 still passes.
        image = torch.rand(1, 3, 16, 16, device=device, dtype=dtype)
        inverted = torch.tensor([[[10.0, 10.0, 9.5, 9.5]]], device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoxFrameWarning)
            bare = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", "bbox_xyxy"])
        bare(image, inverted)
        for key in ("bbox_xyxy_index", "bbox_xyxy_edge"):
            seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=["input", key])
            with pytest.raises(ValueError, match="negative widths"):
                seq(image, inverted)

    @pytest.mark.parametrize("key", ["bbox_xyxy_index", "bbox_xyxy_edge", "keypoints_edge"])
    def test_inverse_round_trip_6710(self, key, device, dtype):
        image = torch.rand(2, 3, 32, 40, device=device, dtype=dtype)
        data = torch.tensor([[[3.0, 5.0, 17.0, 29.0]], [[20.0, 1.0, 39.0, 13.0]]], device=device, dtype=dtype)
        if key.startswith("keypoints"):
            data = data.reshape(2, 2, 2)
        # No rotation: the axis-aligned bounds of a rotated box do not map back to the box.
        aug = K.RandomAffine(degrees=0.0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0)
        seq = K.AugmentationSequential(aug, data_keys=["input", key])
        out = seq(image, data)
        restored = seq.inverse(*out)[1]
        self.assert_close(restored, data, low_tolerance=True)

    @pytest.mark.parametrize("key, far", [("bbox_xyxy_edge", 20.0), ("bbox_xyxy_index", 19.5)])
    def test_convention_standalone_mosaic_window_follows_the_key_frame_6710(self, key, far, device, dtype):
        image = torch.rand(2, 3, 20, 20, device=device, dtype=dtype)
        full = torch.tensor([[[0.0, 0.0, 20.0, 20.0]]], device=device, dtype=dtype).repeat(2, 1, 1)
        mosaic = K.RandomMosaic((20, 20), p=1.0)
        out = mosaic(image, full, data_keys=[DataKey.INPUT, DataKey.get(key)])[1]
        assert out.amax().item() == far
        assert out.amin().item() == (0.0 if key.endswith("edge") else -0.5)

    @pytest.mark.parametrize("key, hflip_x", [("bbox_xyxy_edge", (34.0, 54.0)), ("bbox_xyxy_index", (33.0, 53.0))])
    def test_convention_video_keys_move_corners_in_their_frame_6710(self, key, hflip_x, device, dtype):
        video = torch.rand(1, 2, 3, 64, 64, device=device, dtype=dtype)
        corners = torch.tensor(_CORNERS, device=device, dtype=dtype)[:, None].repeat(1, 2, 1, 1, 1)
        seq = K.AugmentationSequential(K.VideoSequential(K.RandomHorizontalFlip(p=1.0)), data_keys=["input", key])
        out = seq(video, corners)[1]
        x0, x1 = hflip_x
        self.assert_close(out, torch.tensor([[[[x0, 12.0, x1, 40.0]]] * 2], device=device, dtype=dtype))


class TestFrameWarnings:
    @pytest.mark.parametrize("key, replacements", [("bbox_xyxy", "bbox_xyxy_index"), ("bbox", "bbox_index")])
    def test_bare_key_warns_once_and_names_both_replacements(self, key, replacements, fresh_warnings):
        with pytest.warns(BoxFrameWarning, match=replacements) as record:
            K.AugmentationSequential(K.RandomHorizontalFlip(), data_keys=["input", key])
        assert replacements.replace("index", "edge") in str(record[0].message)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BoxFrameWarning)
            K.AugmentationSequential(K.RandomHorizontalFlip(), data_keys=["input", key])

    @pytest.mark.parametrize(
        "keys",
        [
            ["input", "bbox_xyxy_index", "keypoints"],
            ["input", "bbox_xyxy_edge", "keypoints_edge"],
            ["input", "bbox_xywh", "keypoints_edge"],
            ["input", "bbox_edge", "bbox_xyxy_edge", "mask"],
        ],
    )
    def test_frame_named_keys_of_one_frame_do_not_warn(self, keys, fresh_warnings):
        with warnings.catch_warnings():
            warnings.simplefilter("error", BoxFrameWarning)
            K.AugmentationSequential(K.RandomHorizontalFlip(), data_keys=keys)

    @pytest.mark.parametrize(
        "keys",
        [
            ["input", "bbox_xyxy_edge", "keypoints"],
            ["input", "bbox_xywh", "keypoints"],
            ["input", "bbox_xyxy_index", "keypoints_edge"],
        ],
    )
    def test_mixed_frames_warn(self, keys, fresh_warnings):
        with pytest.warns(BoxFrameWarning, match="different frames"):
            K.AugmentationSequential(K.RandomHorizontalFlip(), data_keys=keys)

    def test_dict_keys_resolve_to_frame_named_keys(self, fresh_warnings):
        image = torch.rand(1, 3, 64, 64)
        box = torch.tensor(_BOX)
        seq = K.AugmentationSequential(K.RandomHorizontalFlip(p=1.0), data_keys=None)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BoxFrameWarning)
            out = seq({"image": image, "bbox_xyxy_edge": box, "keypoints_edge": box.reshape(1, 2, 2)})
        assert out["bbox_xyxy_edge"].tolist() == [[[34.0, 12.0, 54.0, 40.0]]]
        assert out["keypoints_edge"].tolist() == [[[54.0, 12.0], [34.0, 40.0]]]
        with pytest.warns(BoxFrameWarning, match="bbox_xyxy_index"):
            seq({"image": image, "bbox_xyxy": box})
