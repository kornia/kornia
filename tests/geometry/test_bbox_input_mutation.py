import torch

from kornia.geometry.bbox import transform_bbox


def test_transform_bbox_xywh_keeps_input():
    boxes = torch.tensor([[1.0, 1.0, 2.0, 1.0]])
    expected = boxes.clone()

    output = transform_bbox(torch.eye(3).unsqueeze(0), boxes, mode="xywh")

    assert torch.equal(output, expected)
    assert torch.equal(boxes, expected)
