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

"""Tests for BboxParams."""

import pytest
import torch

from kornia.augmentations.container import BboxParams, filter_bboxes


def test_construction_defaults():
    bp = BboxParams()
    assert bp.format == "xyxy"
    assert bp.min_area == 0.0
    assert bp.min_visibility == 0.0
    assert bp.label_fields == ()


def test_invalid_format():
    with pytest.raises(ValueError, match="format"):
        BboxParams(format="bad")


def test_invalid_min_visibility():
    with pytest.raises(ValueError, match="min_visibility"):
        BboxParams(min_visibility=1.5)
    with pytest.raises(ValueError, match="min_visibility"):
        BboxParams(min_visibility=-0.1)


def test_invalid_min_area():
    with pytest.raises(ValueError, match="min_area"):
        BboxParams(min_area=-1.0)


def test_filter_no_drops():
    boxes_pre = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    boxes_post = boxes_pre.clone()
    out_boxes, _ = filter_bboxes(boxes_pre, boxes_post, (100, 100), BboxParams())
    assert out_boxes.shape == (2, 4)


def test_filter_drops_below_min_area():
    boxes_pre = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 51.0, 51.0]])
    boxes_post = boxes_pre.clone()
    bp = BboxParams(min_area=10.0)
    out_boxes, _ = filter_bboxes(boxes_pre, boxes_post, (100, 100), bp)
    assert out_boxes.shape == (1, 4)
    assert torch.allclose(out_boxes[0], torch.tensor([0.0, 0.0, 10.0, 10.0]))


def test_filter_drops_below_min_visibility():
    # box 1: post visible only 50% of pre; min_vis=0.6 should drop it
    boxes_pre = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    boxes_post = torch.tensor([[0.0, 0.0, 10.0, 5.0]])  # half height
    bp = BboxParams(min_visibility=0.6)
    out_boxes, _ = filter_bboxes(boxes_pre, boxes_post, (100, 100), bp)
    assert out_boxes.shape == (0, 4)


def test_filter_clamps_to_image_bounds():
    boxes_pre = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    boxes_post = torch.tensor([[-5.0, -5.0, 105.0, 105.0]])  # outside 100x100
    out_boxes, _ = filter_bboxes(boxes_pre, boxes_post, (100, 100), BboxParams())
    assert torch.allclose(out_boxes[0], torch.tensor([0.0, 0.0, 100.0, 100.0]))


def test_filter_label_fields_in_lockstep():
    boxes_pre = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 51.0, 51.0]])
    boxes_post = boxes_pre.clone()
    labels = {"category": torch.tensor([1, 2]), "score": torch.tensor([0.9, 0.5])}
    bp = BboxParams(min_area=10.0, label_fields=("category", "score"))
    out_boxes, out_labels = filter_bboxes(boxes_pre, boxes_post, (100, 100), bp, labels=labels)
    assert out_boxes.shape == (1, 4)
    assert out_labels["category"].tolist() == [1]
    assert torch.isclose(out_labels["score"][0], torch.tensor(0.9))


def test_format_xywh_round_trip():
    bp = BboxParams(format="xywh")
    boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # x, y, w, h
    out_boxes, _ = filter_bboxes(boxes, boxes, (100, 100), bp)
    assert torch.allclose(out_boxes, boxes)


def test_format_cxcywh_round_trip():
    bp = BboxParams(format="cxcywh")
    boxes = torch.tensor([[50.0, 50.0, 20.0, 20.0]])  # cx, cy, w, h
    out_boxes, _ = filter_bboxes(boxes, boxes, (100, 100), bp)
    assert torch.allclose(out_boxes, boxes)
