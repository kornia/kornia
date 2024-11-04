# based on: https://github.com/ShiqiYu/libfacedetection.train/blob/74f3aa77c63234dd954d21286e9a60703b8d0868/tasks/task1/yufacedetectnet.py  # noqa
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.geometry.bbox import nms as nms_kornia

__all__ = ["FaceDetector", "FaceDetectorResult", "FaceKeypoint"]


url: str = "https://github.com/kornia/data/raw/main/yunet_final.pth"


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
        """The [x y] position of a given facial keypoint.

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
    r"""Detect faces in a given image using a CNN.

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
            "name": "YuFaceDetectNet",
            "min_sizes": [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
            "steps": [8, 16, 32, 64],
            "variance": [0.1, 0.2],
            "clip": False,
        }
        self.min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.steps = [8, 16, 32, 64]
        self.variance = [0.1, 0.2]
        self.clip = False
        self.model = YuFaceDetectNet("test", pretrained=True)
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

        priors = _PriorBox(self.min_sizes, self.steps, self.clip, image_size=(height, width))
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


# utils for the network


class ConvDPUnit(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, withBNRelu: bool = True) -> None:
        super().__init__()
        self.add_module("conv1", nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True, groups=1))
        self.add_module("conv2", nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels))
        if withBNRelu:
            self.add_module("bn", nn.BatchNorm2d(out_channels))
            self.add_module("relu", nn.ReLU(inplace=True))


class Conv_head(nn.Sequential):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.add_module("conv1", nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1))
        self.add_module("bn1", nn.BatchNorm2d(mid_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv2", ConvDPUnit(mid_channels, out_channels))


class Conv4layerBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, withBNRelu: bool = True) -> None:
        super().__init__()
        self.add_module("conv1", ConvDPUnit(in_channels, in_channels, True))
        self.add_module("conv2", ConvDPUnit(in_channels, out_channels, withBNRelu))


class YuFaceDetectNet(nn.Module):
    def __init__(self, phase: str, pretrained: bool) -> None:
        super().__init__()
        self.phase = phase
        self.num_classes = 2

        self.model0 = Conv_head(3, 16, 16)
        self.model1 = Conv4layerBlock(16, 64)
        self.model2 = Conv4layerBlock(64, 64)
        self.model3 = Conv4layerBlock(64, 64)
        self.model4 = Conv4layerBlock(64, 64)
        self.model5 = Conv4layerBlock(64, 64)
        self.model6 = Conv4layerBlock(64, 64)

        self.head = nn.Sequential(
            Conv4layerBlock(64, 3 * (14 + 2 + 1), False),
            Conv4layerBlock(64, 2 * (14 + 2 + 1), False),
            Conv4layerBlock(64, 2 * (14 + 2 + 1), False),
            Conv4layerBlock(64, 3 * (14 + 2 + 1), False),
        )

        if self.phase == "train":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(url, map_location=torch.device("cpu"))
            self.load_state_dict(pretrained_dict, strict=True)
        self.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        detection_sources, head_list = [], []

        x = self.model0(x)
        x = F.max_pool2d(x, 2)
        x = self.model1(x)
        x = self.model2(x)
        x = F.max_pool2d(x, 2)
        x = self.model3(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model4(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model5(x)
        detection_sources.append(x)

        x = F.max_pool2d(x, 2)
        x = self.model6(x)
        detection_sources.append(x)

        for i, h in enumerate(self.head):
            x_tmp = h(detection_sources[i])
            head_list.append(x_tmp.permute(0, 2, 3, 1).contiguous())

        head_data = torch.cat([o.view(o.size(0), -1) for o in head_list], 1)
        head_data = head_data.view(head_data.size(0), -1, 17)

        loc_data, conf_data, iou_data = head_data.split((14, 2, 1), dim=-1)

        if self.phase == "test":
            conf_data = torch.softmax(conf_data, dim=-1)
        else:
            loc_data = loc_data.view(loc_data.size(0), -1, 14)
            conf_data = conf_data.view(conf_data.size(0), -1, self.num_classes)
            iou_data = iou_data.view(iou_data.size(0), -1, 1)

        return {"loc": loc_data, "conf": conf_data, "iou": iou_data}


# utils for post-processing


# Adapted from https://github.com/Hakuyume/chainer-ssd
def _decode(loc: torch.Tensor, priors: torch.Tensor, variances: List[float]) -> torch.Tensor:
    """Decode locations from predictions using priors to undo the encoding we did for offset regression at train
    time.

    Args:
        loc:location predictions for loc layers. Shape: [num_priors,4].
        priors: Prior boxes in center-offset form. Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes.

    Return:
        Tensor containing decoded bounding box predictions.
    """
    boxes = torch.cat(
        (
            priors[:, 0:2] + loc[:, 0:2] * variances[0] * priors[:, 2:4],
            priors[:, 2:4] * torch.exp(loc[:, 2:4] * variances[1]),
            priors[:, 0:2] + loc[:, 4:6] * variances[0] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 6:8] * variances[0] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 8:10] * variances[0] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 10:12] * variances[0] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 12:14] * variances[0] * priors[:, 2:4],
        ),
        1,
    )
    # prepare final output
    tmp = boxes[:, 0:2] - boxes[:, 2:4] / 2
    return torch.cat((tmp, boxes[:, 2:4] + tmp, boxes[:, 4:]), dim=-1)


class _PriorBox:
    def __init__(self, min_sizes: List[List[int]], steps: List[int], clip: bool, image_size: Tuple[int, int]) -> None:
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.image_size = image_size

        self.device: torch.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32

        for i in range(4):
            if self.steps[i] != math.pow(2, (i + 3)):
                raise ValueError("steps must be [8,16,32,64]")

        self.feature_map_2th = [int(int((self.image_size[0] + 1) / 2) / 2), int(int((self.image_size[1] + 1) / 2) / 2)]
        self.feature_map_3th = [int(self.feature_map_2th[0] / 2), int(self.feature_map_2th[1] / 2)]
        self.feature_map_4th = [int(self.feature_map_3th[0] / 2), int(self.feature_map_3th[1] / 2)]
        self.feature_map_5th = [int(self.feature_map_4th[0] / 2), int(self.feature_map_4th[1] / 2)]
        self.feature_map_6th = [int(self.feature_map_5th[0] / 2), int(self.feature_map_5th[1] / 2)]

        self.feature_maps = [self.feature_map_3th, self.feature_map_4th, self.feature_map_5th, self.feature_map_6th]

    def to(self, device: torch.device, dtype: torch.dtype) -> "_PriorBox":
        self.device = device
        self.dtype = dtype
        return self

    def __call__(self) -> torch.Tensor:
        anchors: List[float] = []
        for k, f in enumerate(self.feature_maps):
            min_sizes: List[int] = self.min_sizes[k]
            # NOTE: the nested loop it's to make torchscript happy
            for i in range(f[0]):
                for j in range(f[1]):
                    for min_size in min_sizes:
                        s_kx = min_size / self.image_size[1]
                        s_ky = min_size / self.image_size[0]

                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.tensor(anchors, device=self.device, dtype=self.dtype).view(-1, 4)
        if self.clip:
            output = output.clamp(max=1, min=0)
        return output
