from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from kornia.color import rgb_to_grayscale
from kornia.feature.laf import extract_patches_from_pyramid
from kornia.feature.affine_shape import LAFAffNetShapeEstimator
from kornia.feature.hardnet import HardNet
from kornia.feature.laf import get_laf_center, raise_error_if_laf_is_not_valid
from kornia.feature.orientation import LAFOrienter, PassLAF
from kornia.feature.responses import BlobDoG, CornerGFTT
from kornia.feature.scale_space_detector import ScaleSpaceDetector
from kornia.feature.siftdesc import SIFTDescriptor
from kornia.geometry import ConvQuadInterp3d, ScalePyramid


def get_laf_descriptors(img: torch.Tensor,
                        lafs: torch.Tensor,
                        patch_descriptor: nn.Module,
                        patch_size: int = 32,
                        grayscale_descriptor: bool = True) -> torch.Tensor:
    r"""Function to get local descriptors, corresponding to LAFs (keypoints)

    Args:
        img (torch.Tensor): image features with shape [BxCxHxW]
        lafs (torch.Tensor): local affine frames [BxNx2x3]
        patch_descriptor (nn.Module): patch descriptor, e.g. :class:`kornia.feature.SIFTDescriptor` or :class:`kornia.feature.HardNet`  # noqa: E501
        patch_size (int): patch size in pixels, which descriptor expects
        grayscale_descriptor (bool): True if patch_descriptor expects single-channel image

    Returns:
        descriptors (torch.Tensor): local descriptors of shape [BxNxD] where D is descriptor size. """
    raise_error_if_laf_is_not_valid(lafs)
    dev: torch.device = img.device
    dtype: torch.dtype = img.dtype
    assert dev == lafs.device
    patch_descriptor = patch_descriptor.to(dev)
    if grayscale_descriptor and img.size(1) == 3:
        timg = rgb_to_grayscale(img)
    else:
        timg = img
    patch_descriptor.eval()
    patches: torch.Tensor = extract_patches_from_pyramid(timg, lafs, patch_size)
    B, N, CH, H, W = patches.size()
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :)
    descs: torch.Tensor = patch_descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
    return descs


class LAFDescriptor(nn.Module):
    r"""Module to get local descriptors, corresponding to LAFs (keypoints).
    See :func:`~kornia.feature.get_laf_descriptors`

    Args:
        patch_descriptor_module (nn.Module): patch descriptor, e.g.
    :class:`kornia.feature.SIFTDescriptor` or :class:`kornia.feature.HardNet`
        patch_size (int): patch size in pixels, which descriptor expects
        grayscale_descriptor (bool): True if patch_descriptor expects single-channel image

    Returns:
        descriptors (torch.Tensor): local descriptors of shape [BxNxD] where D is descriptor size.
    """

    def __init__(self,
                 patch_descriptor_module: nn.Module = HardNet(True),
                 patch_size: int = 32,
                 grayscale_descriptor: bool = True):
        super().__init__()
        self.descriptor = patch_descriptor_module
        self.patch_size = patch_size
        self.grayscale_descriptor = grayscale_descriptor
        return

    def __repr__(self):
        return self.__class__.__name__ + '(' + \
            'descriptor=' + self.descriptor.__repr__() + ', ' + \
            'patch_size=' + str(self.patch_size) + ', ' + \
            'grayscale_descriptor=' + str(self.grayscale_descriptor) + ')'

    def forward(self, img: torch.Tensor, lafs: torch.Tensor) -> torch.Tensor:
        r"""Three stage local feature detection.

        First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.

        Args:
            img (torch.Tensor): image features with shape [BxCxHxW]
            lafs (torch.Tensor): local affine frames [BxNx2x3]

        Returns:
            descriptors (torch.Tensor): local descriptors of shape [BxNxD] where D is descriptor size.
        """
        return get_laf_descriptors(img, lafs, self.descriptor, self.patch_size, self.grayscale_descriptor)


class LocalFeature(nn.Module):
    """Module, which combines local feature detector and descriptor, (see
    :class:`kornia.feature.ScaleSpaceDetector` see :class:`kornia.feature.LAFDescriptor`)"""
    def __init__(self,
                 detector: ScaleSpaceDetector,
                 descriptor: LAFDescriptor):
        super().__init__()
        self.detector = detector
        self.descriptor = descriptor

    def forward(self,
                img: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
                                                              torch.Tensor,
                                                              torch.Tensor]:  # type: ignore
        """
        Args:
            img (torch.Tensor): image to extract features with shape [BxCxHxW]
            mask (torch.Tensor, optional): a mask with weights where to apply the
            response function. The shape must be the same as the input image.
        Returns:
            lafs (torch.Tensor): shape [BxNx2x3]. Detected local affine frames.
            responses (torch.Tensor): shape [BxNx1]. Response function values for corresponding lafs
            descriptors (torch.Tensor): shape [BxNxD]  local descriptors of shape [BxNxD] where D is descriptor size
            """
        lafs, responses = self.detector(img, mask)
        descs = self.descriptor(img, lafs)
        return lafs, responses, descs


class SIFTFeature(LocalFeature):
    """Convenience module, which implements DoG detector + (Root)SIFT descriptor.
    Still not as good as OpenCV/VLFeat because of https://github.com/kornia/kornia/pull/884, but we are working on it
    """
    def __init__(self,
                 num_features: int = 8000,
                 upright: bool = False,
                 rootsift: bool = True,
                 device: torch.device = torch.device('cpu')):
        super(LocalFeature, self).__init__()
        self.patch_size: int = 41
        self.detector = ScaleSpaceDetector(num_features,
                                           resp_module=BlobDoG(),
                                           nms_module=ConvQuadInterp3d(10),
                                           scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=True),
                                           ori_module=PassLAF() if upright else LAFOrienter(19),
                                           scale_space_response=True,
                                           minima_are_also_good=True,
                                           mr_size=6.0).to(device)
        self.descriptor = LAFDescriptor(SIFTDescriptor(patch_size=self.patch_size, rootsift=rootsift),
                                        patch_size=self.patch_size,
                                        grayscale_descriptor=True).to(device)
        return


class GFTTAffNetHardNet(LocalFeature):
    """Convenience module, which implements GFTT detector + AffNet-HardNet descriptor."""
    def __init__(self,
                 num_features: int = 8000,
                 upright: bool = False,
                 device: torch.device = torch.device('cpu')):
        super(LocalFeature, self).__init__()
        self.patch_size: int = 32
        self.detector = ScaleSpaceDetector(num_features,
                                           resp_module=CornerGFTT(),
                                           nms_module=ConvQuadInterp3d(10, 1e-5),
                                           scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=False),
                                           ori_module=PassLAF() if upright else LAFOrienter(19),
                                           aff_module=LAFAffNetShapeEstimator(True).eval(),
                                           mr_size=6.0).to(device)
        self.descriptor = LAFDescriptor(HardNet(True).eval(),
                                        patch_size=32,
                                        grayscale_descriptor=True).to(device)
        return


class LocalFeatureMatcher(nn.Module):
    r"""Module, which finds correspondences between two images based on local features.
    Args:
        detector: Local feature detector. See :class:`~kornia.feature.ScaleSpaceDetector`.
        descriptor: Local patch descriptor, see :class:`~kornia.feature.HardNet`
                    or :class:`~kornia.feature.SIFTDescriptor`.
        matcher: Descriptor matcher, see :class:`~kornia.feature.DescriptorMatcher`.

    Returns:
        Dictionary with image correspondences and confidence scores.

    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> gftt_hardnet_matcher = LocalFeatureMatcher(GFTTAffNetHardNet(10), kornia.feature.DescriptorMatcher('snn', 0.8))  # noqa: E501
        >>> out = gftt_hardnet_matcher(input)
    """

    def __init__(self, local_feature, matcher):
        super().__init__()
        self.local_feature = local_feature
        self.matcher = matcher
        self.eval()

    def extract_features(self,
                         image: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Function for feature extraction from simple image."""
        lafs0, resps0, descs0 = self.local_feature(image, mask)
        return {"lafs": lafs0, "responses": resps0, "descriptors": descs0}

    def no_match_output(self, device: torch.device, dtype: torch.dtype):
        return {'keypoints0': torch.empty(0, 2, device=device, dtype=dtype),
                'keypoints1': torch.empty(0, 2, device=device, dtype=dtype),
                'lafs0': torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
                'lafs1': torch.empty(0, 0, 2, 3, device=device, dtype=dtype),
                'confidence': torch.empty(0, device=device, dtype=dtype),
                'batch_indexes': torch.empty(0, device=device, dtype=torch.long)}

    def forward(self, data: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            data: {
                'image0': (torch.Tensor): (N, 1, H1, W1)
                'image1': (torch.Tensor): (N, 1, H2, W2)
                'mask0'(optional) : (torch.Tensor): (N, H1, W1) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H2, W2)
            }

        Returns:
            out: {
                    "keypoints0": (torch.Tensor): (NC, 2) matching keypoints from image0
                    "keypoints1":  (torch.Tensor): (NC, 2) matching keypoints from image1
                    "confidence": (torch.Tensor): (NC) - confidence score [0, 1]
                    "keypoints0": (torch.Tensor): (1, NC, 2, 3) matching LAFs from image0
                    "keypoints1":  (torch.Tensor): (1, NC, 2, 3) matching LAFs from image1
                    "batch_indexes": (torch.Tensor): (NC) - batch indexes for the keypoints and lafs
            }
        """
        num_image_pairs = data['image0'].shape[0]
        if ('lafs0' not in data.keys()) or ('descriptors0' not in data.keys()):
            # One can supply pre-extracted local features
            feats_dict0 = self.extract_features(data['image0'])
            lafs0, descs0 = feats_dict0['lafs'], feats_dict0['descriptors']
        else:
            lafs0, descs0 = data['lafs0'], data['descriptors0']
        if ('lafs1' not in data.keys()) or ('descriptors1' not in data.keys()):
            feats_dict1 = self.extract_features(data['image1'])
            lafs1, descs1 = feats_dict1['lafs'], feats_dict1['descriptors']
        else:
            lafs1, descs1 = data['lafs1'], data['descriptors1']
        keypoints0 = get_laf_center(lafs0)
        keypoints1 = get_laf_center(lafs1)

        out_keypoints0: List[torch.Tensor] = []
        out_keypoints1: List[torch.Tensor] = []
        out_confidence: List[torch.Tensor] = []
        out_batch_indexes: List[torch.Tensor] = []
        out_lafs0: List[torch.Tensor] = []
        out_lafs1: List[torch.Tensor] = []

        for batch_idx in range(num_image_pairs):
            dists, idxs = self.matcher(descs0[batch_idx], descs1[batch_idx])
            if len(idxs) == 0:
                continue
            current_keypoints_0 = keypoints0[batch_idx, idxs[:, 0]]
            current_keypoints_1 = keypoints1[batch_idx, idxs[:, 1]]
            current_lafs_0 = lafs0[batch_idx, idxs[:, 0]]
            current_lafs_1 = lafs1[batch_idx, idxs[:, 1]]

            out_confidence.append(1.0 - dists)
            batch_idxs = batch_idx * torch.ones(len(dists),
                                                device=keypoints0.device,
                                                dtype=torch.long)
            out_keypoints0.append(current_keypoints_0)
            out_keypoints1.append(current_keypoints_1)
            out_lafs0.append(current_lafs_0)
            out_lafs1.append(current_lafs_1)
            out_batch_indexes.append(batch_idxs)
        if len(out_batch_indexes) == 0:
            return self.no_match_output(data['image0'].device,
                                        data['image0'].dtype)
        out = {'keypoints0': torch.cat(out_keypoints0, dim=0).view(-1, 2),
               'keypoints1': torch.cat(out_keypoints1, dim=0).view(-1, 2),
               'lafs0': torch.cat(out_lafs0, dim=0).view(1, -1, 2, 3),
               'lafs1': torch.cat(out_lafs1, dim=0).view(1, -1, 2, 3),
               'confidence': torch.cat(out_confidence, dim=0).view(-1),
               'batch_indexes': torch.cat(out_batch_indexes, dim=0).view(-1)}
        return out
