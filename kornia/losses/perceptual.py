from typing import Callable, Tuple

import torch
import torch.nn as nn
import torchvision

import kornia
from kornia import testing
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_perspective


class LossNetwork(nn.Module):

    def __init__(
        self,
        model_name: str = 'resnet34',
        output_layer: int = 1,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if 'resnet' not in model_name:
            raise ValueError('Only ResNets are supported ATM')

        # Define resnet model
        loss_network_fn = getattr(torchvision.models, model_name)
        self.resnet = loss_network_fn(pretrained=True, progress=False)

        # Clear unnecessary layers
        self.output_layer = output_layer
        if self.output_layer < 2:
            self.resnet.layer2 = torch.nn.Identity()
        if self.output_layer < 3:
            self.resnet.layer3 = torch.nn.Identity()
        if self.output_layer < 4:
            self.resnet.layer4 = torch.nn.Identity()
        self.resnet.avgpool = torch.nn.Identity()
        self.resnet.fc = torch.nn.Identity()

        # Freeze the model
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)

        if self.output_layer > 1:
            x = self.resnet.layer2(x)
        if self.output_layer > 2:
            x = self.resnet.layer3(x)
        if self.output_layer > 3:
            x = self.resnet.layer4(x)

        return x


def _image_shape_to_corners(image: torch.Tensor) -> torch.Tensor:
    """Convert image size to 4 corners representation in clockwise order.

    Args:
        image: image tensor with shape :math:`(B, C, H, W)` where B = batch size,
            C = number of channels

    Return:
        the corners of the image.
    """
    testing.KORNIA_CHECK_SHAPE(image, ['*', '*', '*', '*'])
    batch_size = image.shape[0]
    image_width = image.shape[-2]
    image_height = image.shape[-1]
    corners = torch.tensor(
        [[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]],
        device=image.device,
        dtype=image.dtype,
        requires_grad=False,
    )
    corners = corners.repeat(batch_size, 1, 1)

    return corners


def _four_point_to_homography(corners: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Convert 4-point representation introduced in :cite:`detone2016deep` to homography.

    Args:
        corners: corners tensor with shape :math:`(B, 4, 2)` where B = batch size
        deltas: deltas tensor with shape :math:`(B, 4, 2)` where B = batch size

    Return:
        the converted homography.
    """

    testing.KORNIA_CHECK_SHAPE(deltas, ['*', '4', '2'])
    testing.KORNIA_CHECK_SHAPE(corners, ['*', '4', '2'])
    testing.KORNIA_CHECK(
        corners.size(0) == deltas.size(0),
        f'Expected corners batch_size ({corners.size(0)}) to match deltas batch '
        f'size ({deltas.size(0)}).'
    )

    corners_hat = corners + deltas
    homography_inv = get_perspective_transform(corners, corners_hat)
    homography = torch.inverse(homography_inv)
    return homography


def _warp(image: torch.Tensor, delta_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert 4-point representation introduced in :cite:`detone2016deep` to homography.

    Args:
        image: image tensor with shape :math:`(B, C, H, W)` where B = batch size,
            C = number of channels
        deltas: deltas tensor with shape :math:`(B, 4, 2)` where B = batch size

    Return:
        the warped images.
    """
    corners = _image_shape_to_corners(image=image)
    homography = _four_point_to_homography(corners=corners, deltas=delta_hat)
    image_warped = warp_perspective(image, homography, (image.shape[-2], image.shape[-1]))
    return image_warped, torch.inverse(homography)


def bihome_loss(
    patch_1: torch.Tensor,
    patch_2: torch.Tensor,
    delta_hat_12: torch.Tensor,
    delta_hat_21: torch.Tensor,
    triplet_mu: float,
    loss_network: nn.Module,
) -> torch.Tensor:
    r"""biHomE loss implementation.

    Based on: :cite:`koguciuk2021perceptual` and https://github.com/NeurAI-Lab/biHomE.

    Args:
        patch_1: image tensor with shape :math:`(B, C, H, W)` where B = batch size,
            C = number of classes
        patch_2: image tensor with shape :math:`(B, C, H, W)` where B = batch size,
            C = number of classes
        delta_hat_12: predicted corner differences from image 1 to image 2 with shape
            :math:`(B, 4, 2)`, where B = batch size.
        delta_hat_21: predicted corner differences from image 2 to image 1 with shape
            :math:`(B, 4, 2)`, where B = batch size.
        triplet_mu: Homography matrix regularization weight.
        loss_network: loss network used.

    Return:
        the computed loss.
    """
    testing.KORNIA_CHECK_SHAPE(
        patch_1,
        ['*', '*', '*', '*'],
        # f"Invalid input shape of patch_1, we expect BxCxHxW. Got: {patch_1.shape}",
    )
    testing.KORNIA_CHECK_SHAPE(
        patch_2,
        ['*', '*', '*', '*'],
        # f"Invalid input shape of patch_2, we expect BxCxHxW. Got: {patch_2.shape}",
    )
    testing.KORNIA_CHECK(
        patch_1.shape == patch_2.shape,
        f'Expected patch_1 shape ({patch_1.shape}) to match patch_2 shape ({patch_2.shape}).',
    )
    testing.KORNIA_CHECK_SHAPE(
        delta_hat_12,
        ['*', '4', '2'],
        # f"Invalid input shape of delta_hat_12, we expect Bx4x2. Got: {delta_hat_12.shape}",
    )
    testing.KORNIA_CHECK(
        delta_hat_12.size(0) == patch_1.size(0),
        f'Expected delta_hat_12 batch_size ({delta_hat_12.size(0)}) to match patch_1 batch size '
        f'({patch_1.size(0)}).',
    )
    testing.KORNIA_CHECK_SHAPE(
        delta_hat_21,
        ['*', '4', '2'],
        # f"Invalid input shape of delta_hat_21, we expect Bx4x2. Got: {delta_hat_21.shape}",
    )
    testing.KORNIA_CHECK(
        delta_hat_21.size(0) == patch_1.size(0),
        f'Expected delta_hat_21 batch_size ({delta_hat_21.size(0)}) to match patch_1 batch size '
        f'({patch_1.size(0)}).',
    )
    testing.KORNIA_CHECK(
        isinstance(loss_network, nn.Module),
        f"loss_network type is not a str. Got {type(loss_network)}",
    )

    # Compute features of both patches
    patch_1_f = loss_network(patch_1)
    patch_2_f = loss_network(patch_2)

    # Warp patch 1 with delta hat_12
    patch_1_prime, h1 = _warp(patch_1, delta_hat=delta_hat_12)
    patch_1_prime_f = loss_network(patch_1_prime)

    # Warp patch 2 with delta hat_21
    patch_2_prime, h2 = _warp(patch_2, delta_hat=delta_hat_21)
    patch_2_prime_f = loss_network(patch_2_prime)

    # Create and warp masks
    patch_1_m = torch.ones_like(patch_1)
    patch_2_m = torch.ones_like(patch_2)
    patch_1_m_prime, _ = _warp(patch_1_m, delta_hat=delta_hat_12)
    patch_2_m_prime, _ = _warp(patch_2_m, delta_hat=delta_hat_21)

    # Mask size mismatch downsampling
    _, _, f_h, _ = patch_1_prime_f.shape
    downsample_factor = patch_1_m.shape[-1] // f_h
    patch_1_m = torch.squeeze(
        torch.nn.functional.avg_pool2d(
            input=patch_1_m,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
        ), dim=1
    )
    patch_2_m = torch.squeeze(
        torch.nn.functional.avg_pool2d(
            input=patch_2_m,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
        ), dim=1
    )
    patch_1_m_prime = torch.squeeze(
        torch.nn.functional.avg_pool2d(
            input=patch_1_m_prime,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
        ), dim=1
    )
    patch_2_m_prime = torch.squeeze(
        torch.nn.functional.avg_pool2d(
            input=patch_2_m_prime,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
        ), dim=1
    )

    # Triplet Margin Loss
    l1 = torch.sum(torch.abs(patch_1_prime_f - patch_2_f), dim=1)
    l2 = torch.sum(torch.abs(patch_1_f - patch_2_prime_f), dim=1)
    l3 = torch.sum(torch.abs(patch_1_f - patch_2_f), dim=1)
    ln1_nom = torch.sum(torch.sum(patch_1_m_prime * patch_2_m * (l1 - l3), dim=-1), dim=-1)
    ln1_den = torch.sum(torch.sum(patch_1_m_prime * patch_2_m, dim=-1), dim=-1)
    ln1_den = torch.max(ln1_den, torch.ones_like(ln1_den))
    ln2_nom = torch.sum(torch.sum(patch_1_m * patch_2_m_prime * (l2 - l3), dim=-1), dim=-1)
    ln2_den = torch.sum(torch.sum(patch_1_m * patch_2_m_prime, dim=-1), dim=-1)
    ln2_den = torch.max(ln2_den, torch.ones_like(ln2_den))
    ln1 = torch.sum(ln1_nom / ln1_den)
    ln2 = torch.sum(ln2_nom / ln2_den)

    # Regularization
    eye = kornia.eye_like(3, h1)
    ln3 = torch.sum((torch.matmul(h1, h2) - eye) ** 2) * triplet_mu

    loss = ln1 + ln2 + ln3
    return loss


class BiHomELoss(nn.Module):
    r"""Criterion that computes biHomE perceptual loss.

    Based on: :cite:`koguciuk2021perceptual` and https://github.com/NeurAI-Lab/biHomE.

    Args:
        loss_network_name: loss network name from torchvision models.
        loss_network: the user can use its own Loss Network implementation instead of predefined
            in torchvision.
        triplet_mu: Homography matrix regularization weight.
    """

    def __init__(
        self,
        loss_network_name: str = 'resnet34',
        loss_network: Callable = None,
        triplet_mu: float = 0.01,
    ) -> None:

        super().__init__()

        if loss_network is None and loss_network_name is None:
            raise RuntimeError("At least one should be defined out of: loss_network and loss_network_name")

        if loss_network is not None and loss_network_name is not None:
            raise RuntimeWarning("Both are defined out of: loss_network and loss_network_name - I will use "
                                 "loss_network_name")

        if loss_network_name is not None:
            self.loss_network = LossNetwork(loss_network_name)
        else:
            self.loss_network = loss_network

        self.triplet_mu = triplet_mu

    def forward(
        self,
        patch_1: torch.Tensor,
        patch_2: torch.Tensor,
        delta_hat_12: torch.Tensor,
        delta_hat_21: torch.Tensor,
    ) -> torch.Tensor:
        loss = bihome_loss(
            patch_1,
            patch_2,
            delta_hat_12,
            delta_hat_21,
            self.triplet_mu,
            loss_network=self.loss_network
        )
        return loss
