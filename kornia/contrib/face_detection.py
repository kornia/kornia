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

from enum import Enum
from typing import Dict, List, Optional

import torch
from torch import nn

from kornia.geometry.bbox import nms as nms_kornia
from kornia.models.yunet import YuNet
from kornia.models.yunet.processors import PriorBox
from kornia.models.yunet.processors import decode as _decode

__all__ = ["FaceDetector", "FaceDetectorResult", "FaceKeypoint"]


class FaceKeypoint(Enum):
    r"""Define the keypoints detected in a face.

    The left/right convention is based on the screen viewer.
    """

    EYE_LEFT = 0
    EYE_RIGHT = 1
    NOSE = 2
    MOUTH_LEFT = 3
    MOUTH_RIGHT = 4


class FaceDetectorResult:
    r"""Encapsulate the results obtained by the :py:class:`kornia.contrib.FaceDetector`.

    Args:
        data: the encoded results coming from the feature detector with shape :math:`(14,)`.

    """

    def __init__(self, data: torch.Tensor) -> None:
        if len(data) < 15:
            raise ValueError(f"Result must comes as vector of size(14). Got: {data.shape}.")
        self._data = data

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "FaceDetectorResult":
        """Like :func:`torch.nn.Module.to()` method."""
        self._data = self._data.to(device=device, dtype=dtype)
        return self

    @property
    def xmin(self) -> torch.Tensor:
        """The bounding box top-left x-coordinate."""
        return self._data[..., 0]

    @property
    def ymin(self) -> torch.Tensor:
        """The bounding box top-left y-coordinate."""
        return self._data[..., 1]

    @property
    def xmax(self) -> torch.Tensor:
        """The bounding box bottom-right x-coordinate."""
        return self._data[..., 2]

    @property
    def ymax(self) -> torch.Tensor:
        """The bounding box bottom-right y-coordinate."""
        return self._data[..., 3]

    def get_keypoint(self, keypoint: FaceKeypoint) -> torch.Tensor:
        """Get the [x y] position of a given facial keypoint.

        Args:
            keypoint: the keypoint type to return the position.

        """
        if keypoint == FaceKeypoint.EYE_LEFT:
            out = self._data[..., (4, 5)]
        elif keypoint == FaceKeypoint.EYE_RIGHT:
            out = self._data[..., (6, 7)]
        elif keypoint == FaceKeypoint.NOSE:
            out = self._data[..., (8, 9)]
        elif keypoint == FaceKeypoint.MOUTH_LEFT:
            out = self._data[..., (10, 11)]
        elif keypoint == FaceKeypoint.MOUTH_RIGHT:
            out = self._data[..., (12, 13)]
        else:
            raise ValueError(f"Not valid keypoint type. Got: {keypoint}.")
        return out

    @property
    def score(self) -> torch.Tensor:
        """The detection score."""
        return self._data[..., 14]

    @property
    def width(self) -> torch.Tensor:
        """The bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self) -> torch.Tensor:
        """The bounding box height."""
        return self.ymax - self.ymin

    @property
    def top_left(self) -> torch.Tensor:
        """The [x y] position of the top-left coordinate of the bounding box."""
        return self._data[..., (0, 1)]

    @property
    def top_right(self) -> torch.Tensor:
        """The [x y] position of the top-left coordinate of the bounding box."""
        out = self.top_left
        out[..., 0] += self.width
        return out

    @property
    def bottom_right(self) -> torch.Tensor:
        """The [x y] position of the bottom-right coordinate of the bounding box."""
        return self._data[..., (2, 3)]

    @property
    def bottom_left(self) -> torch.Tensor:
        """The [x y] position of the top-left coordinate of the bounding box."""
        out = self.top_left
        out[..., 1] += self.height
        return out


class FaceDetector(nn.Module):
    r"""Detect faces in a given image using YuNet model.

    This is a high-level API that wraps the :py:class:`kornia.models.YuNet` model for face detection.
    By default, it uses the method described in :cite:`facedetect-yu`.

    Args:
        top_k: the maximum number of detections to return before the nms.
        confidence_threshold: the threshold used to discard detections.
        nms_threshold: the threshold used by the nms for iou.
        keep_top_k: the maximum number of detections to return after the nms.

    Return:
        A list of B tensors with shape :math:`(N,15)` to be used with :py:class:`kornia.contrib.FaceDetectorResult`.

    Example:
        >>> img = torch.rand(1, 3, 320, 320)
        >>> detect = FaceDetector()
        >>> res = detect(img)

    """

    def __init__(
        self, top_k: int = 5000, confidence_threshold: float = 0.3, nms_threshold: float = 0.3, keep_top_k: int = 750
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.config = {
            "name": "YuNet",
            "min_sizes": [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
            "steps": [8, 16, 32, 64],
            "variance": [0.1, 0.2],
            "clip": False,
        }
        self.min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.steps = [8, 16, 32, 64]
        self.variance = [0.1, 0.2]
        self.clip = False
        self.model = YuNet("test", pretrained=True)
        self.nms = nms_kornia

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def postprocess(self, data: Dict[str, torch.Tensor], height: int, width: int) -> List[torch.Tensor]:
        loc, conf, iou = data["loc"], data["conf"], data["iou"]

        scale = torch.tensor(
            [width, height, width, height, width, height, width, height, width, height, width, height, width, height],
            device=loc.device,
            dtype=loc.dtype,
        )  # 14

        priors = PriorBox(self.min_sizes, self.steps, self.clip, image_size=(height, width))
        priors = priors.to(loc.device, loc.dtype)

        batched_dets: List[torch.Tensor] = []
        for batch_elem in range(loc.shape[0]):
            boxes = _decode(loc[batch_elem], priors(), self.variance)  # Nx14
            boxes = boxes * scale

            # clamp here for the compatibility for ONNX
            cls_scores, iou_scores = conf[batch_elem, :, 1], iou[batch_elem, :, 0]
            scores = (cls_scores * iou_scores.clamp(0.0, 1.0)).sqrt()

            # ignore low scores
            inds = scores > self.confidence_threshold
            boxes, scores = boxes[inds], scores[inds]

            # keep top-K before NMS
            order = scores.sort(descending=True)[1][: self.top_k]
            boxes, scores = boxes[order], scores[order]

            # performd NMS
            # NOTE: nms need to be revise since does not export well to onnx
            dets = torch.cat((boxes, scores[:, None]), dim=-1)  # Nx15
            keep = self.nms(boxes[:, :4], scores, self.nms_threshold)
            if len(keep) > 0:
                dets = dets[keep, :]

            # keep top-K faster NMS
            batched_dets.append(dets[: self.keep_top_k])
        return batched_dets

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        r"""Detect faces in a given batch of images.

        Args:
            image: batch of images :math:`(B,3,H,W)`

        Return:
            List[torch.Tensor]: list with the boxes found on each image. :math:`Bx(N,15)`.

        """
        img = self.preprocess(image)
        out = self.model(img)
        return self.postprocess(out, img.shape[-2], img.shape[-1])
