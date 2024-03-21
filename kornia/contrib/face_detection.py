# based on: https://github.com/ShiqiYu/libfacedetection.train/blob/74f3aa77c63234dd954d21286e9a60703b8d0868/tasks/task1/yufacedetectnet.py  # noqa
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from kornia.geometry.bbox import nms as nms_kornia
from kornia.utils.helpers import map_location_to_cpu

__all__ = ["FaceDetector", "FaceDetectorResult", "FaceKeypoint"]


url: str = "https://github.com/ShiqiYu/libfacedetection.train/raw/master/weights/yunet_n.pth"


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
        config = {
            "name": "YuFaceDetectNet",
            "backbone": {
                "stage_channels": [
                    (3, 16, 16),
                    (16, 64),
                    (64, 64),
                    (64, 64),
                    (64, 64),
                    (64, 64),
                ],
                "sample_layer_idxs": [3, 4, 5],
                "downsample_layer_idxs": [0, 2, 3, 4],
            },
            "neck": {"in_channels": [64, 64, 64], "out_idx": [0, 1, 2]},
            "bbox_head": {
                "in_channels": 64,
                "feat_channels": 64,
                "stacked_convs": 0,
                "shared_stacked_convs": 1,
            },
            "steps": [8, 16, 32, 64],
            "clip": False,
        }
        self.model = YuFaceDetectNet(config, "test", pretrained=True)
        self.steps = [8, 16, 32]
        self.clip = False
        self.nms = nms_kornia

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def postprocess(self, data: Dict[str, torch.Tensor], processed_dims, org_dims) -> List[torch.Tensor]:
        height, width = processed_dims
        org_height, org_width = org_dims
        bbox_preds, cls_preds, obj_preds, kps_preds = (
            data["bbox_preds"],
            data["cls_preds"],
            data["obj_preds"],
            data["kps_preds"],
        )
        num_imgs = cls_preds[0].shape[0]
        featmap_sizes = [cls_pred.shape[2:] for cls_pred in cls_preds]

        priors = _PriorBox(self.steps, (height, width), featmap_sizes)()

        flatten_cls_preds = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1) for cls_pred in cls_preds]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
        flatten_obj_preds = [obj_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1) for obj_pred in obj_preds]
        flatten_kps_preds = [kps_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 10) for kps_pred in kps_preds]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_obj_preds = torch.cat(flatten_obj_preds, dim=1).sigmoid()
        flatten_kps_preds = torch.cat(flatten_kps_preds, dim=1)
        flatten_priros = torch.cat(priors)

        scale = torch.tensor(
            [
                org_width / width,
                org_height / height,
                org_width / width,
                org_height / height,
                org_width / width,
                org_height / height,
                org_width / width,
                org_height / height,
                org_width / width,
                org_height / height,
                org_width / width,
                org_height / height,
                org_width / width,
                org_height / height,
            ],
            device=flatten_bbox_preds.device,
            dtype=flatten_bbox_preds.dtype,
        )

        flatten_priros = flatten_priros.to(flatten_bbox_preds.device, flatten_bbox_preds.dtype)
        batched_dets: List[torch.Tensor] = []
        for batch_elem in range(flatten_bbox_preds.shape[0]):
            boxes = _decode(torch.cat([flatten_bbox_preds, flatten_kps_preds], -1)[batch_elem], flatten_priros)  # Nx14

            boxes *= scale
            cls_scores = flatten_cls_preds[batch_elem]
            score_factor = flatten_obj_preds[batch_elem]

            # ignore low scores
            max_scores, labels = torch.max(cls_scores, 1)
            valid_mask = score_factor * max_scores >= self.confidence_threshold
            boxes = boxes[valid_mask]
            scores = max_scores[valid_mask] * score_factor[valid_mask]
            labels = labels[valid_mask]

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
        return self.postprocess(out, (img.shape[-2], img.shape[-1]), (image.shape[-2], image.shape[-1]))


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


class TFPN(nn.Module):
    def __init__(self, in_channels: List, out_idx: int) -> None:
        super().__init__()
        self.num_layers = len(in_channels)
        self.out_idx = out_idx
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.lateral_convs.append(ConvDPUnit(in_channels[i], in_channels[i], True))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        num_feats = len(feats)
        # top-down flow
        for i in range(num_feats - 1, 0, -1):
            feats[i] = self.lateral_convs[i](feats[i])
            feats[i - 1] = feats[i - 1] + F.interpolate(feats[i], scale_factor=2.0, mode="nearest")
        feats[0] = self.lateral_convs[0](feats[0])
        outs = [feats[i] for i in self.out_idx]
        return outs


class MultiLevelShareConvs(nn.Module):
    def __init__(self, in_channels: int, feat_channels: int, shared_stacked_convs: int, strides) -> None:
        super().__init__()
        self.shared_stacked_convs = shared_stacked_convs
        for i, _ in enumerate(strides):
            single_level_share_convs = []
            for j in range(self.shared_stacked_convs):
                chn = in_channels if j == 0 else feat_channels
                single_level_share_convs.append(ConvDPUnit(chn, feat_channels))
            self.add_module(f"{i}", nn.Sequential(*single_level_share_convs))


class YuFaceDetectNet(nn.Module):
    def __init__(self, config: Dict, phase: str = "train", pretrained: bool = False) -> None:
        super().__init__()
        self.stage_channels = config["backbone"]["stage_channels"]
        self.phase = phase
        self.num_classes = 1
        self.num_layers = len(self.stage_channels)
        self.num_keypoints = 5
        self.sample_layer_idxs = config["backbone"]["sample_layer_idxs"]
        self.downsample_layer_idxs = config["backbone"]["downsample_layer_idxs"]
        self.stacked_convs = config["bbox_head"]["stacked_convs"]
        self.shared_stacked_convs = config["bbox_head"]["shared_stacked_convs"]
        self.strides = [8, 16, 32]

        self.backbone = nn.Sequential()
        self.backbone.add_module("model0", Conv_head(*self.stage_channels[0]))
        for i in range(1, self.num_layers):
            self.backbone.add_module(f"model{i}", Conv4layerBlock(*self.stage_channels[i]))

        self.neck = TFPN(config["neck"]["in_channels"], config["neck"]["out_idx"])

        self.bbox_head = nn.ModuleDict()

        self.bbox_head["multi_level_cls"] = nn.Sequential(
            ConvDPUnit(64, self.num_classes, False),
            ConvDPUnit(64, self.num_classes, False),
            ConvDPUnit(64, self.num_classes, False),
        )

        self.bbox_head["multi_level_bbox"] = nn.Sequential(
            ConvDPUnit(64, 4, False),
            ConvDPUnit(64, 4, False),
            ConvDPUnit(64, 4, False),
        )

        self.bbox_head["multi_level_obj"] = nn.Sequential(
            ConvDPUnit(64, 1, False),
            ConvDPUnit(64, 1, False),
            ConvDPUnit(64, 1, False),
        )

        self.bbox_head["multi_level_kps"] = nn.Sequential(
            ConvDPUnit(64, self.num_keypoints * 2, False),
            ConvDPUnit(64, self.num_keypoints * 2, False),
            ConvDPUnit(64, self.num_keypoints * 2, False),
        )

        self.bbox_head["multi_level_share_convs"] = MultiLevelShareConvs(
            config["bbox_head"]["in_channels"],
            config["bbox_head"]["feat_channels"],
            self.shared_stacked_convs,
            self.strides,
        )

        # use torch.hub to load pretrained model
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(url, map_location=map_location_to_cpu)
            self.load_state_dict(pretrained_dict["state_dict"], strict=True)
        self.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        detection_sources = []

        for i in range(self.num_layers):
            # x = getattr(self, "model{}".format(i))(x)
            x = self.backbone[i](x)
            if i in self.sample_layer_idxs:
                detection_sources.append(x)
            if i in self.downsample_layer_idxs:
                x = F.max_pool2d(x, 2)

        detection_sources = self.neck(detection_sources)

        if self.shared_stacked_convs > 0:
            feats = [
                getattr(self.bbox_head["multi_level_share_convs"], str(i))(feat)
                for i, feat in zip(range(len(self.strides)), detection_sources)
            ]

        cls_preds = [conv(feat) for conv, feat in zip(self.bbox_head["multi_level_cls"], feats)]

        bbox_preds = [conv(feat) for conv, feat in zip(self.bbox_head["multi_level_bbox"], feats)]

        obj_preds = [conv(feat) for conv, feat in zip(self.bbox_head["multi_level_obj"], feats)]

        kps_preds = [conv(feat) for conv, feat in zip(self.bbox_head["multi_level_kps"], feats)]

        return {"cls_preds": cls_preds, "bbox_preds": bbox_preds, "obj_preds": obj_preds, "kps_preds": kps_preds}


# utils for post-processing


# Adapted from https://github.com/Hakuyume/chainer-ssd
def _decode(loc: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
    """Decode locations from predictions using priors to undo the encoding we did for offset regression at train
    time.

    Args:
        loc:location predictions for loc layers. Shape: [num_priors,4].
        priors: Prior boxes in center-offset form. Shape: [num_priors,4].

    Return:
        Tensor containing decoded bounding box predictions.
    """
    boxes = torch.cat(
        (
            priors[:, 0:2] + loc[:, 0:2] * priors[:, 2:4],
            priors[:, 2:4] * torch.exp(loc[:, 2:4]),
            priors[:, 0:2] + loc[:, 4:6] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 6:8] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 8:10] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 10:12] * priors[:, 2:4],
            priors[:, 0:2] + loc[:, 12:14] * priors[:, 2:4],
        ),
        1,
    )
    # prepare final output
    tmp = boxes[:, 0:2] - boxes[:, 2:4] / 2
    return torch.cat((tmp, boxes[:, 2:4] + tmp, boxes[:, 4:]), dim=-1)


class _PriorBox:
    def __init__(
        self,
        strides: List[int],
        image_size: Tuple[int, int],
        featmap_sizes: List[Tuple[int, int]],
        offset: float = 0.0,
    ):
        self.strides = strides
        self.offset = offset
        self.image_size = image_size
        self.featmap_sizes = featmap_sizes

        self.device = torch.device("cpu")
        self.dtype = torch.float32

    def to(self, device: torch.device, dtype: torch.dtype) -> "_PriorBox":
        self.device = device
        self.dtype = dtype
        return self

    def __call__(self) -> List[torch.Tensor]:
        all_points = []
        for level, (feat_h, feat_w) in enumerate(self.featmap_sizes):
            stride = self.strides[level]

            # Generating grid for feature map
            shifts_x = (torch.arange(0, feat_w, device=self.device) + self.offset) * stride
            shifts_y = (torch.arange(0, feat_h, device=self.device) + self.offset) * stride

            # Creating a meshgrid
            shift_xx, shift_yy = torch.meshgrid(shifts_x, shifts_y, indexing="xy")
            shifts_xx = shift_xx.reshape(-1)
            shifts_yy = shift_yy.reshape(-1)

            # Include stride information if required
            strides_xx = torch.full_like(shifts_xx, stride, dtype=self.dtype, device=self.device)
            strides_yy = torch.full_like(shifts_yy, stride, dtype=self.dtype, device=self.device)
            points = torch.stack([shifts_xx, shifts_yy, strides_xx, strides_yy], dim=1)

            all_points.append(points)

        return all_points
