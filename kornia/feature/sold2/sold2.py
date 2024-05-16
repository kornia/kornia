import warnings
from typing import Any, Dict, Optional

import torch

from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.feature.sold2.structures import DetectorCfg
from kornia.matching import WunschLineMatcher
from kornia.utils import dataclass_to_dict, dict_to_dataclass, map_location_to_cpu

from .backbones import SOLD2Net
from .sold2_detector import LineSegmentDetectionModule, line_map_to_segments, prob_to_junctions

urls: Dict[str, str] = {}
urls["wireframe"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/sold2_wireframe.pth"


class SOLD2(Module):
    r"""Module, which detects and describe line segments in an image.

    This is based on the original code from the paper "SOLD²: Self-supervised
    Occlusion-aware Line Detector and Descriptor". See :cite:`SOLD22021` for more details.

    Args:
        config: Dict specifying parameters. None will load the default parameters,
            which are tuned for images in the range 400~800 px.
        pretrained: If True, download and set pretrained weights to the model.

    Returns:
        The raw junction and line heatmaps, the semi-dense descriptor map,
        as well as the list of detected line segments (ij coordinates convention).

    Example:
        >>> images = torch.rand(2, 1, 512, 512)
        >>> sold2 = SOLD2()
        >>> outputs = sold2(images)
        >>> line_seg1 = outputs["line_segments"][0]
        >>> line_seg2 = outputs["line_segments"][1]
        >>> desc1 = outputs["dense_desc"][0]
        >>> desc2 = outputs["dense_desc"][1]
        >>> matches = sold2.match(line_seg1, line_seg2, desc1[None], desc2[None])
    """

    def __init__(self, pretrained: bool = True, config: Optional[DetectorCfg] = None) -> None:
        if isinstance(config, dict):
            warnings.warn(
                "Usage of config as a plain dictionary is deprecated in favor of"
                " `kornia.features.sold2.structures.DetectorCfg`. The support of plain dictionaries"
                "as config will be removed in kornia v0.8.0 (December 2024).",
                category=DeprecationWarning,
                stacklevel=2,
            )
            config = dict_to_dataclass(config, DetectorCfg)
        super().__init__()
        # Initialize some parameters
        self.config = config if config is not None else DetectorCfg()
        self.config.use_descriptor = True  # Only difference to SOLD2_detector DetectorCfg
        self.grid_size = self.config.grid_size
        self.junc_detect_thresh = self.config.detection_thresh
        self.max_num_junctions = self.config.max_num_junctions

        # Load the pre-trained model
        self.model = SOLD2Net(dataclass_to_dict(self.config))
        if pretrained:
            pretrained_dict = torch.hub.load_state_dict_from_url(urls["wireframe"], map_location=map_location_to_cpu)
            state_dict = self.adapt_state_dict(pretrained_dict["model_state_dict"])
            self.model.load_state_dict(state_dict)
        self.eval()

        # Initialize the line detector
        self.line_detector = LineSegmentDetectionModule(self.config.line_detector_cfg)

        # Initialize the line matcher
        self.line_matcher = WunschLineMatcher(self.config.line_matcher_cfg)

    def forward(self, img: Tensor) -> Dict[str, Any]:
        """
        Args:
            img: batched images with shape :math:`(B, 1, H, W)`.

        Return:
            - ``line_segments``: list of N line segments in each of the B images :math:`List[(N, 2, 2)]`.
            - ``junction_heatmap``: raw junction heatmap of shape :math:`(B, H, W)`.
            - ``line_heatmap``: raw line heatmap of shape :math:`(B, H, W)`.
            - ``dense_desc``: the semi-dense descriptor map of shape :math:`(B, 128, H/4, W/4)`.
        """
        KORNIA_CHECK_SHAPE(img, ["B", "1", "H", "W"])
        outputs = {}

        # Forward pass of the CNN backbone
        net_outputs = self.model(img)
        outputs["junction_heatmap"] = net_outputs["junctions"]
        outputs["line_heatmap"] = net_outputs["heatmap"]
        outputs["dense_desc"] = net_outputs["descriptors"]

        # Loop through all images
        lines = []
        for junc_prob, heatmap in zip(net_outputs["junctions"], net_outputs["heatmap"]):
            # Get the junctions
            junctions = prob_to_junctions(junc_prob, self.grid_size, self.junc_detect_thresh, self.max_num_junctions)

            # Run the line detector
            line_map, junctions, _ = self.line_detector.detect(junctions, heatmap)
            lines.append(line_map_to_segments(junctions, line_map))
        outputs["line_segments"] = lines

        return outputs

    def match(self, line_seg1: Tensor, line_seg2: Tensor, desc1: Tensor, desc2: Tensor) -> Tensor:
        """Find the best matches between two sets of line segments and their corresponding descriptors.

        Args:
            line_seg1, line_seg2: list of line segments in two images, with shape [num_lines, 2, 2].
            desc1, desc2: semi-dense descriptor maps of the images, with shape [1, 128, H/4, W/4].
        Returns:
            A np.array of size [num_lines1] indicating the index in line_seg2 of the matched line,
            for each line in line_seg1. -1 means that the line is not matched.
        """
        return self.line_matcher(line_seg1, line_seg2, desc1, desc2)

    def adapt_state_dict(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        del state_dict["w_junc"]
        del state_dict["w_heatmap"]
        del state_dict["w_desc"]
        state_dict["heatmap_decoder.conv_block_lst.2.0.weight"] = state_dict["heatmap_decoder.conv_block_lst.2.weight"]
        state_dict["heatmap_decoder.conv_block_lst.2.0.bias"] = state_dict["heatmap_decoder.conv_block_lst.2.bias"]
        del state_dict["heatmap_decoder.conv_block_lst.2.weight"]
        del state_dict["heatmap_decoder.conv_block_lst.2.bias"]
        return state_dict
