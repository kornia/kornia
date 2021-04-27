from typing import Tuple, List, Optional

import torch
import torch.nn as nn

from kornia.feature.laf import raise_error_if_laf_is_not_valid
from kornia.feature import ScaleSpaceDetector, HardNet, SIFTDescriptor, BlobDoG, LAFOrienter, PassLAF
from kornia.feature import extract_patches_from_pyramid
from kornia.geometry import ScalePyramid, ConvQuadInterp3d
from kornia.color import rgb_to_grayscale


def get_laf_descriptors(img: torch.Tensor,
                        lafs: torch.Tensor,
                        patch_descriptor: nn.Module,
                        patch_size: int = 32,
                        grayscale_descriptor: bool = True) -> torch.Tensor:
    """Function to get local descriptors, corresponding to LAFs (keypoints)
        Args:
            img (torch.Tensor): image features with shape [BxCxHxW]
            lafs (torch.Tensor): local affine frames [BxNx2x3]
            patch_descriptor (nn.Module): patch descriptor,
        e.g. :class:`kornia.feature.SIFTDescriptor` or :class:`kornia.feature.HardNet`
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
    """Module to get local descriptors, corresponding to LAFs (keypoints). See :func:`~kornia.feature.get_laf_descriptors`
        Args:
            patch_descriptor_module (nn.Module): patch descriptor, e.g.
        :class:`kornia.feature.SIFTDescriptor` or :class:`kornia.feature.HardNet`
            patch_size (int): patch size in pixels, which descriptor expects
            grayscale_descriptor (bool): True if patch_descriptor expects single-channel image
        Returns:
            descriptors (torch.Tensor): local descriptors of shape [BxNxD] where D is descriptor size. """

    def __init__(self,
                 patch_descriptor_module: nn.Module = HardNet(True),
                 patch_size: int = 32,
                 grayscale_descriptor: bool = True):
        super(LAFDescriptor, self).__init__()
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
        """Three stage local feature detection. First the location and scale of interest points are determined by
        detect function. Then affine shape and orientation.
        Args:
            img (torch.Tensor): image features with shape [BxCxHxW]
            lafs (torch.Tensor): local affine frames [BxNx2x3]
        Returns:
            descriptors (torch.Tensor): local descriptors of shape [BxNxD] where D is descriptor size. """
        return get_laf_descriptors(img, lafs, self.descriptor, self.patch_size, self.grayscale_descriptor)


class LocalFeature(nn.Module):
    """Module, which combines local feature detector and descriptor, (see :class:`kornia.feature.ScaleSpaceDetector`
        see :class:`kornia.feature.LAFDescriptor`)"""
    def __init__(self,
                 detector: ScaleSpaceDetector,
                 descriptor: LAFDescriptor):
        super(LocalFeature, self).__init__()
        self.detector = detector
        self.descriptor = descriptor
        return

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
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
    Still not as good as OpenCV/VLFeat because of
    https://github.com/kornia/kornia/pull/884, but we are working on it"""
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
