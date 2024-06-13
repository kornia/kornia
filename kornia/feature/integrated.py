import warnings
from typing import ClassVar, Dict, List, Optional, Tuple

import torch

from kornia.color import rgb_to_grayscale
from kornia.constants import pi
from kornia.core import Device, Module, Tensor, concatenate, deg2rad
from kornia.core.check import KORNIA_CHECK_LAF
from kornia.geometry.subpix import ConvQuadInterp3d
from kornia.geometry.transform import ScalePyramid

from .affine_shape import LAFAffNetShapeEstimator
from .hardnet import HardNet
from .keynet import KeyNetDetector
from .laf import extract_patches_from_pyramid, get_laf_center, get_laf_orientation, get_laf_scale, scale_laf
from .lightglue import LightGlue
from .matching import GeometryAwareDescriptorMatcher, _no_match
from .orientation import LAFOrienter, OriNet, PassLAF
from .responses import BlobDoG, BlobDoGSingle, BlobHessian, CornerGFTT
from .scale_space_detector import (
    Detector_config,
    MultiResolutionDetector,
    ScaleSpaceDetector,
    get_default_detector_config,
)
from .siftdesc import SIFTDescriptor


def get_laf_descriptors(
    img: Tensor, lafs: Tensor, patch_descriptor: Module, patch_size: int = 32, grayscale_descriptor: bool = True
) -> Tensor:
    r"""Function to get local descriptors, corresponding to LAFs (keypoints).

    Args:
        img: image features with shape :math:`(B,C,H,W)`.
        lafs: local affine frames :math:`(B,N,2,3)`.
        patch_descriptor: patch descriptor module, e.g. :class:`~kornia.feature.SIFTDescriptor`
            or :class:`~kornia.feature.HardNet`.
        patch_size: patch size in pixels, which descriptor expects.
        grayscale_descriptor: True if ``patch_descriptor`` expects single-channel image.

    Returns:
        Local descriptors of shape :math:`(B,N,D)` where :math:`D` is descriptor size.
    """
    KORNIA_CHECK_LAF(lafs)
    patch_descriptor = patch_descriptor.to(img)
    patch_descriptor.eval()

    timg: Tensor = img
    if lafs.shape[1] == 0:
        warnings.warn(f"LAF contains no keypoints {lafs.shape}, returning empty tensor")
        return torch.empty(lafs.shape[0], lafs.shape[1], 128, dtype=lafs.dtype, device=lafs.device)
    if grayscale_descriptor and img.size(1) == 3:
        timg = rgb_to_grayscale(img)

    patches: Tensor = extract_patches_from_pyramid(timg, lafs, patch_size)
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :)
    B, N, CH, H, W = patches.size()
    return patch_descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)


class LAFDescriptor(Module):
    r"""Module to get local descriptors, corresponding to LAFs (keypoints).

    Internally uses :func:`~kornia.feature.get_laf_descriptors`.

    Args:
        patch_descriptor_module: patch descriptor module, e.g. :class:`~kornia.feature.SIFTDescriptor`
            or :class:`~kornia.feature.HardNet`. Default: :class:`~kornia.feature.HardNet`.
        patch_size: patch size in pixels, which descriptor expects.
        grayscale_descriptor: ``True`` if patch_descriptor expects single-channel image.
    """

    def __init__(
        self, patch_descriptor_module: Optional[Module] = None, patch_size: int = 32, grayscale_descriptor: bool = True
    ) -> None:
        super().__init__()
        if patch_descriptor_module is None:
            patch_descriptor_module = HardNet(True)
        self.descriptor = patch_descriptor_module
        self.patch_size = patch_size
        self.grayscale_descriptor = grayscale_descriptor

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(descriptor={self.descriptor.__repr__()}, "
            f"patch_size={self.patch_size}, "
            f"grayscale_descriptor='{self.grayscale_descriptor})"
        )

    def forward(self, img: Tensor, lafs: Tensor) -> Tensor:
        r"""Three stage local feature detection.

        First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.

        Args:
            img: image features with shape :math:`(B,C,H,W)`.
            lafs: local affine frames :math:`(B,N,2,3)`.

        Returns:
            Local descriptors of shape :math:`(B,N,D)` where :math:`D` is descriptor size.
        """
        return get_laf_descriptors(img, lafs, self.descriptor, self.patch_size, self.grayscale_descriptor)


class LocalFeature(Module):
    """Module, which combines local feature detector and descriptor.

    Args:
        detector: the detection module.
        descriptor: the descriptor module.
        scaling_coef: multiplier for change default detector scale (e.g. it is too small for KeyNet by default)
    """

    def __init__(self, detector: Module, descriptor: LAFDescriptor, scaling_coef: float = 1.0) -> None:
        super().__init__()
        self.detector = detector
        self.descriptor = descriptor
        if scaling_coef <= 0:
            raise ValueError(f"Scaling coef should be >= 0, got {scaling_coef}")
        self.scaling_coef = scaling_coef

    def forward(self, img: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            img: image to extract features with shape :math:`(B,C,H,W)`.
            mask: a mask with weights where to apply the response function.
                The shape must be the same as the input image.

        Returns:
            - Detected local affine frames with shape :math:`(B,N,2,3)`.
            - Response function values for corresponding lafs with shape :math:`(B,N,1)`.
            - Local descriptors of shape :math:`(B,N,D)` where :math:`D` is descriptor size.
        """
        lafs, responses = self.detector(img, mask)
        lafs = scale_laf(lafs, self.scaling_coef)
        descs = self.descriptor(img, lafs)
        return (lafs, responses, descs)


class SIFTFeature(LocalFeature):
    """Convenience module, which implements DoG detector + (Root)SIFT descriptor.

    Using `kornia.feature.MultiResolutionDetector` without blur pyramid Still not as good as OpenCV/VLFeat because of
    https://github.com/kornia/kornia/pull/884,
    but we are working on it
    """

    def __init__(
        self,
        num_features: int = 8000,
        upright: bool = False,
        rootsift: bool = True,
        device: Device = torch.device("cpu"),
        config: Detector_config = get_default_detector_config(),
    ) -> None:
        patch_size: int = 41
        detector = MultiResolutionDetector(
            BlobDoGSingle(1.0, 1.6),
            num_features,
            config,
            ori_module=PassLAF() if upright else LAFOrienter(19),
            aff_module=PassLAF(),
        ).to(device)
        descriptor = LAFDescriptor(
            SIFTDescriptor(patch_size=patch_size, rootsift=rootsift), patch_size=patch_size, grayscale_descriptor=True
        ).to(device)
        super().__init__(detector, descriptor)


class SIFTFeatureScaleSpace(LocalFeature):
    """Convenience module, which implements DoG detector + (Root)SIFT descriptor. Using
    `kornia.feature.ScaleSpaceDetector` with blur pyramid.

    Still not as good as OpenCV/VLFeat because of https://github.com/kornia/kornia/pull/884, but we are working on it
    """

    def __init__(
        self,
        num_features: int = 8000,
        upright: bool = False,
        rootsift: bool = True,
        device: Device = torch.device("cpu"),
    ) -> None:
        patch_size: int = 41
        detector = ScaleSpaceDetector(
            num_features,
            resp_module=BlobDoG(),
            nms_module=ConvQuadInterp3d(10),
            scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=True),
            ori_module=PassLAF() if upright else LAFOrienter(19),
            scale_space_response=True,
            minima_are_also_good=True,
            mr_size=6.0,
        ).to(device)
        descriptor = LAFDescriptor(
            SIFTDescriptor(patch_size=patch_size, rootsift=rootsift), patch_size=patch_size, grayscale_descriptor=True
        ).to(device)
        super().__init__(detector, descriptor)


class GFTTAffNetHardNet(LocalFeature):
    """Convenience module, which implements GFTT detector + AffNet-HardNet descriptor."""

    def __init__(
        self,
        num_features: int = 8000,
        upright: bool = False,
        device: Device = torch.device("cpu"),
        config: Detector_config = get_default_detector_config(),
    ) -> None:
        detector = MultiResolutionDetector(
            CornerGFTT(),
            num_features,
            config,
            ori_module=PassLAF() if upright else LAFOrienter(19),
            aff_module=LAFAffNetShapeEstimator(True).eval(),
        ).to(device)
        descriptor = LAFDescriptor(None, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor)


class HesAffNetHardNet(LocalFeature):
    """Convenience module, which implements GFTT detector + AffNet-HardNet descriptor."""

    def __init__(
        self,
        num_features: int = 2048,
        upright: bool = False,
        device: Device = torch.device("cpu"),
        config: Detector_config = get_default_detector_config(),
    ) -> None:
        detector = MultiResolutionDetector(
            BlobHessian(),
            num_features,
            config,
            ori_module=PassLAF() if upright else LAFOrienter(19),
            aff_module=LAFAffNetShapeEstimator(True).eval(),
        ).to(device)
        descriptor = LAFDescriptor(None, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor)


class KeyNetHardNet(LocalFeature):
    """Convenience module, which implements KeyNet detector + HardNet descriptor."""

    def __init__(
        self,
        num_features: int = 8000,
        upright: bool = False,
        device: Device = torch.device("cpu"),
        scale_laf: float = 1.0,
    ) -> None:
        ori_module = PassLAF() if upright else LAFOrienter(angle_detector=OriNet(True))
        detector = KeyNetDetector(True, num_features=num_features, ori_module=ori_module).to(device)
        descriptor = LAFDescriptor(None, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)


class KeyNetAffNetHardNet(LocalFeature):
    """Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.

    .. image:: _static/img/keynet_affnet.jpg
    """

    def __init__(
        self,
        num_features: int = 8000,
        upright: bool = False,
        device: Device = torch.device("cpu"),
        scale_laf: float = 1.0,
    ) -> None:
        ori_module = PassLAF() if upright else LAFOrienter(angle_detector=OriNet(True))
        detector = KeyNetDetector(
            True, num_features=num_features, ori_module=ori_module, aff_module=LAFAffNetShapeEstimator(True).eval()
        ).to(device)
        descriptor = LAFDescriptor(None, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)


class LocalFeatureMatcher(Module):
    r"""Module, which finds correspondences between two images based on local features.

    Args:
        local_feature: Local feature detector. See :class:`~kornia.feature.GFTTAffNetHardNet`.
        matcher: Descriptor matcher, see :class:`~kornia.feature.DescriptorMatcher`.

    Returns:
        Dict[str, Tensor]: Dictionary with image correspondences and confidence scores.

    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> gftt_hardnet_matcher = LocalFeatureMatcher(
        ...     GFTTAffNetHardNet(10), kornia.feature.DescriptorMatcher('snn', 0.8)
        ... )
        >>> out = gftt_hardnet_matcher(input)
    """

    def __init__(self, local_feature: Module, matcher: Module) -> None:
        super().__init__()
        self.local_feature = local_feature
        self.matcher = matcher
        self.eval()

    def extract_features(self, image: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Function for feature extraction from simple image."""
        lafs0, resps0, descs0 = self.local_feature(image, mask)
        return {"lafs": lafs0, "responses": resps0, "descriptors": descs0}

    def no_match_output(self, device: Device, dtype: torch.dtype) -> Dict[str, Tensor]:
        return {
            "keypoints0": torch.empty(0, 2, device=device, dtype=dtype),
            "keypoints1": torch.empty(0, 2, device=device, dtype=dtype),
            "lafs0": torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
            "lafs1": torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
            "confidence": torch.empty(0, device=device, dtype=dtype),
            "batch_indexes": torch.empty(0, device=device, dtype=torch.long),
        }

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            data: dictionary containing the input data in the following format:

        Keyword Args:
            image0: left image with shape :math:`(N, 1, H1, W1)`.
            image1: right image with shape :math:`(N, 1, H2, W2)`.
            mask0 (optional): left image mask. '0' indicates a padded position :math:`(N, H1, W1)`.
            mask1 (optional): right image mask. '0' indicates a padded position :math:`(N, H2, W2)`.

        Returns:
            - ``keypoints0``, matching keypoints from image0 :math:`(NC, 2)`.
            - ``keypoints1``, matching keypoints from image1 :math:`(NC, 2)`.
            - ``confidence``, confidence score [0, 1] :math:`(NC)`.
            - ``lafs0``, matching LAFs from image0 :math:`(1, NC, 2, 3)`.
            - ``lafs1``, matching LAFs from image1 :math:`(1, NC, 2, 3)`.
            - ``batch_indexes``, batch indexes for the keypoints and lafs :math:`(NC)`.
        """
        num_image_pairs: int = data["image0"].shape[0]

        if ("lafs0" not in data.keys()) or ("descriptors0" not in data.keys()):
            # One can supply pre-extracted local features
            feats_dict0: Dict[str, Tensor] = self.extract_features(data["image0"])
            lafs0, descs0 = feats_dict0["lafs"], feats_dict0["descriptors"]
        else:
            lafs0, descs0 = data["lafs0"], data["descriptors0"]

        if ("lafs1" not in data.keys()) or ("descriptors1" not in data.keys()):
            feats_dict1: Dict[str, Tensor] = self.extract_features(data["image1"])
            lafs1, descs1 = feats_dict1["lafs"], feats_dict1["descriptors"]
        else:
            lafs1, descs1 = data["lafs1"], data["descriptors1"]

        keypoints0: Tensor = get_laf_center(lafs0)
        keypoints1: Tensor = get_laf_center(lafs1)

        out_keypoints0: List[Tensor] = []
        out_keypoints1: List[Tensor] = []
        out_confidence: List[Tensor] = []
        out_batch_indexes: List[Tensor] = []
        out_lafs0: List[Tensor] = []
        out_lafs1: List[Tensor] = []

        for batch_idx in range(num_image_pairs):
            dists, idxs = self.matcher(descs0[batch_idx], descs1[batch_idx])
            if len(idxs) == 0:
                continue

            current_keypoints_0 = keypoints0[batch_idx, idxs[:, 0]]
            current_keypoints_1 = keypoints1[batch_idx, idxs[:, 1]]
            current_lafs_0 = lafs0[batch_idx, idxs[:, 0]]
            current_lafs_1 = lafs1[batch_idx, idxs[:, 1]]

            out_confidence.append(1.0 - dists)
            batch_idxs = batch_idx * torch.ones(len(dists), device=keypoints0.device, dtype=torch.long)
            out_keypoints0.append(current_keypoints_0)
            out_keypoints1.append(current_keypoints_1)
            out_lafs0.append(current_lafs_0)
            out_lafs1.append(current_lafs_1)
            out_batch_indexes.append(batch_idxs)

        if len(out_batch_indexes) == 0:
            return self.no_match_output(data["image0"].device, data["image0"].dtype)

        return {
            "keypoints0": concatenate(out_keypoints0, dim=0).view(-1, 2),
            "keypoints1": concatenate(out_keypoints1, dim=0).view(-1, 2),
            "lafs0": concatenate(out_lafs0, dim=0).view(1, -1, 2, 3),
            "lafs1": concatenate(out_lafs1, dim=0).view(1, -1, 2, 3),
            "confidence": concatenate(out_confidence, dim=0).view(-1),
            "batch_indexes": concatenate(out_batch_indexes, dim=0).view(-1),
        }


class LightGlueMatcher(GeometryAwareDescriptorMatcher):
    """LightGlue-based matcher in kornia API.

    This is based on the original code from paper "LightGlue: Local Feature Matching at Light Speed".
    See :cite:`LightGlue2023` for more details.

    Args:
        feature_name: type of feature for matching, can be `disk` or `superpoint`.
        params: LightGlue params.
    """

    known_modes: ClassVar[List[str]] = [
        "aliked",
        "dedodeb",
        "dedodeg",
        "disk",
        "dog_affnet_hardnet",
        "doghardnet",
        "keynet_affnet_hardnet",
        "sift",
        "superpoint",
    ]

    def __init__(self, feature_name: str = "disk", params: Dict = {}) -> None:  # type: ignore
        feature_name_: str = feature_name.lower()
        super().__init__(feature_name_)
        self.feature_name = feature_name_
        self.params = params
        self.matcher = LightGlue(self.feature_name, **params)

    def forward(
        self,
        desc1: Tensor,
        desc2: Tensor,
        lafs1: Tensor,
        lafs2: Tensor,
        hw1: Optional[Tuple[int, int]] = None,
        hw2: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
            lafs1: LAFs of a shape :math:`(1, B1, 2, 3)`.
            lafs2: LAFs of a shape :math:`(1, B2, 2, 3)`.

        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.
        """
        if (desc1.shape[0] < 2) or (desc2.shape[0] < 2):
            return _no_match(desc1)
        keypoints1 = get_laf_center(lafs1)
        keypoints2 = get_laf_center(lafs2)
        if len(desc1.shape) == 2:
            desc1 = desc1.unsqueeze(0)
        if len(desc2.shape) == 2:
            desc2 = desc2.unsqueeze(0)
        dev = lafs1.device
        if hw1 is None:
            hw1_ = keypoints1.max(dim=1)[0].squeeze().flip(0)
        else:
            hw1_ = torch.tensor(hw1, device=dev)
        if hw2 is None:
            hw2_ = keypoints2.max(dim=1)[0].squeeze().flip(0)
        else:
            hw2_ = torch.tensor(hw2, device=dev)
        ori0 = deg2rad(get_laf_orientation(lafs1).reshape(1, -1))
        ori0[ori0 < 0] += 2.0 * pi
        ori1 = deg2rad(get_laf_orientation(lafs2).reshape(1, -1))
        ori1[ori1 < 0] += 2.0 * pi
        input_dict = {
            "image0": {
                "keypoints": keypoints1,
                "scales": get_laf_scale(lafs1).reshape(1, -1),
                "oris": ori0,
                "lafs": lafs1,
                "descriptors": desc1,
                "image_size": hw1_.flip(0).reshape(-1, 2).to(dev),
            },
            "image1": {
                "keypoints": keypoints2,
                "lafs": lafs2,
                "scales": get_laf_scale(lafs2).reshape(1, -1),
                "oris": ori1,
                "descriptors": desc2,
                "image_size": hw2_.flip(0).reshape(-1, 2).to(dev),
            },
        }
        pred = self.matcher(input_dict)
        matches0, mscores0 = pred["matches0"], pred["matching_scores0"]
        valid = matches0 > -1
        matches = torch.stack([torch.where(valid)[1], matches0[valid]], -1)
        return mscores0[valid].reshape(-1, 1), matches
