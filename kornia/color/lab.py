import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

"""
# Adapted from https://github.com/xahidbuffon/rgb-lab-conv/blob/master/rgb_lab_formulation.py and based on:
https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
https://github.com/cameronfabbri/Colorizing-Images-Using-Adversarial-Networks
"""

import sys
sprintf = lambda format, *values : format % values
printf  = lambda format, *values : sys.stdout.write(format % values)

# keep only constants in buffer
class RgbToLab(nn.Module):
    r"""Convert image from RGB to LAB
    The image data is assumed to be in the range of (0, 1).
    args:
        image (torch.Tensor): RGB image to be converted to LAB.
    returns:
        torch.tensor: LAB version of the image.
    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Examples::
        >>> input = torch.rand(2, 3, 4, 5)
        >>> lab = kornia.color.RgbToLab()
        >>> output = lab(input)  # 2x3x4x5
    Reference::
        [1]
    """

    def __init__(self) -> None:
        super(RgbToLab, self).__init__()
        eps = 10e-12
        self.rgb_to_xyz = torch.tensor([
            #    X        Y          Z
            [0.412453, 0.212671, 0.019334], # R
            [0.357580, 0.715160, 0.119193], # G
            [0.180423, 0.072169, 0.950227], # B
        ]) + eps
        self.register_buffer('srgb2xyz_const', self.rgb_to_xyz)

        self.fxfyfz_to_lab = torch.tensor([
            #  l       a       b
            [  0.0,  500.0,    0.0], # fx
            [116.0, -500.0,  200.0], # fy
            [  0.0,    0.0, -200.0], # fz
        ]) + eps
        self.register_buffer('xyz2lab_const', self.fxfyfz_to_lab)

        # normalize for D65 white point
        self.xyz_normalization_const = torch.tensor([1/0.950456, 1.0, 1/1.088754]) + eps
        self.register_buffer('xyz2lab_xyz_normalization_const', self.xyz_normalization_const)

        self.lab_normalization_const = torch.tensor([-16.0, 0.0, 0.0]) + eps
        self.register_buffer('xyz2lab_lab_normalization_const', self.lab_normalization_const)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # with torch.no_grad():
        return self.rgb_to_lab(input)

    # based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
    def rgb_to_lab(self, input: torch.Tensor) -> torch.Tensor:
        r"""Convert an RGB image to LAB
        The image data is assumed to be in the range of (0, 1).
        Args:
            input (torch.Tensor): RGB Image to be converted to LAB.
        Returns:
            torch.Tensor: LAB version of the image.
        See :class:`~kornia.color.RgbToLab` for details."""
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {type(input)}")

        if not(len(input.shape) == 3 or len(input.shape) == 4):
            raise ValueError(f"Input size must have a shape of (*, 3, H, W) or (3, H, W). Got {input.shape}")

        if input.shape[-3] != 3:
            raise ValueError(f"Expected input to have 3 channels, got {input.shape[-3]}")

        input_dims = len(input.shape)
        if input_dims == 3:
            input = input[None, :, :, :]

        # [b, c, h, w] -> [b, h, w, c]
        permuted_input = input.permute(0, 2, 3, 1)
        # self.register_buffer('srgb2lab_permuted_input', permuted_input)
        srgb_pixels = permuted_input.reshape(-1, 3)
        # self.register_buffer('srgb2lab_reshaped', srgb_pixels)
        #     printf("srgb size: %s\n", srgb_pixels.size())
        xyz_pixels = self.srgb_to_xyz(srgb_pixels)
        # self.register_buffer('srgb2xyz_xyz', xyz_pixels)
        lab_pixels = self.xyz_to_cielab(xyz_pixels)
        # self.register_buffer('xyz2lab_lab_', lab_pixels)

        # Reshape to input size
        # [b, h, w, c] -> [b, c, h, w]
        lab_pixels = lab_pixels.reshape(permuted_input.size())
        # self.register_buffer('srgb2lab_lab_reshaped', lab_pixels)
        lab_pixels = lab_pixels.permute(0, 3, 1, 2)
        # self.register_buffer('srgb2lab_lab_permuted', lab_pixels)
        # lab_pixels = lab_pixels.float()
        # self.register_buffer('srgb2lab_lab', lab_pixels)

        if input_dims == 3:
            lab_pixels = lab_pixels[0, :, :, :]

        return lab_pixels

    def srgb_to_xyz(self, input: torch.Tensor) -> torch.Tensor:
        eps = 10e-12

        # Constant tensor used for conversion
        rgb_to_xyz = self.rgb_to_xyz.type(input.data.type())

        linear_mask = (input <= 0.04045).type(input.data.type()) + eps
        # self.register_buffer('srgb2xyz_linear_mask', linear_mask)
        exponential_mask = (input > 0.04045).type(input.data.type()) + eps
        # self.register_buffer('srgb2xyz_exponential_mask', exponential_mask)

        # rgb_pixels[rgb_pixels.ne(rgb_pixels)] = 0
        # rgb_pixels += 10e-12

        # avoid a slightly negative number messing up the conversion
        input = torch.clamp(input, 0.0, 1.0-eps) + eps

        rgb_pixels = (input / 12.92 * linear_mask)
        # temp = ((input + 0.055) / 1.055)
        # temp[temp.ne(temp)] = 0
        # temp += 10e-12
        # temp2 = (temp ** 2.4)
        # temp2[temp2.ne(temp2)] = 0
        # temp2 += 10e-12
        rgb_pixels += (((input + 0.055) / 1.055) ** 2.4)  * exponential_mask

        # self.register_buffer('srgb2xyz_rgb', rgb_pixels)

        xyz_pixels = torch.matmul(rgb_pixels, rgb_to_xyz)
        return xyz_pixels

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    def xyz_to_cielab(self, input: torch.Tensor) -> torch.Tensor:
        eps = 10e-12

        # Constant tensors used for conversion
        fxfyfz_to_lab = self.fxfyfz_to_lab.type(input.data.type())
        xyz_normalization_const = self.xyz_normalization_const.type(input.data.type())
        lab_normalization_const = self.lab_normalization_const.type(input.data.type())

        # normalize for D65 white point
        xyz_normalized_pixels = torch.mul(input, xyz_normalization_const) + eps
        # self.register_buffer('xyz2lab_xyz_normalized', xyz_normalized_pixels)

        epsilon = 6/29
        linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(input.data.type()) + eps
        # self.register_buffer('xyz2lab_linear_mask', linear_mask)

        exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(input.data.type()) + eps
        # self.register_buffer('xyz2lab_exponential_mask', exponential_mask)


        fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask
        ##### TODO: problematic grad here!
        # Function 'PowBackward0' returned nan values in its 0th output.
        xyz_normalized_pixels[xyz_normalized_pixels.ne(xyz_normalized_pixels)] = 0
        xyz_normalized_pixels += eps
        fxfyfz_pixels += (xyz_normalized_pixels ** (1/3)) * exponential_mask

        # eps = 10e-12
        # printf("fxfyfz_pixels: %s\n", (torch.isnan(fxfyfz_pixels) == True).nonzero())
        # printf("fxfyfz_pixels != fxfyfz_pixels: %s\n", (fxfyfz_pixels != fxfyfz_pixels).any())
        # printf("fxfyfz_pixels %s\n", fxfyfz_pixels[(torch.isnan(fxfyfz_pixels) == True)])
        # fxfyfz_pixels[(torch.isnan(fxfyfz_pixels) == True)] = 0
        # printf("fxfyfz_pixels %s\n", fxfyfz_pixels[(torch.isnan(fxfyfz_pixels) == True)])
        # fxfyfz_pixels[fxfyfz_pixels.ne(fxfyfz_pixels)] = 0

        # assert ((input != input).any()), "Checkpoint 0"


        # self.register_buffer('xyz2lab_fxfyfz', fxfyfz_pixels)

        lab_pixels = torch.matmul(fxfyfz_pixels, fxfyfz_to_lab) + lab_normalization_const

        return lab_pixels





class LabToRgb(nn.Module):
    r"""Convert image from LAB to RGB
    The image data is assumed to be in the range of (0, 1).
    args:
        image (torch.Tensor): LAB image to be converted to RGB.
    returns:
        torch.tensor: RGB version of the image.
    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Examples::
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = kornia.color.LabToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def __init__(self) -> None:
        super(LabToRgb, self).__init__()
        eps = 10e-12
        # Constant tensor used for conversion
        self.lab_to_fxfyfz = torch.tensor([
            #   fx      fy        fz
            [1/116.0, 1/116.0,  1/116.0], # l
            [1/500.0,     0.0,      0.0], # a
            [    0.0,     0.0, -1/200.0], # b
        ]) + eps
        self.register_buffer('lab2xyz_const', self.lab_to_fxfyfz)

        self.fxfyfz_normalization_const = torch.tensor([16.0, 0.0, 0.0]) + eps
        self.register_buffer('lab2xyz_fxfyfz_normalization_const', self.fxfyfz_normalization_const)

        self.xyz_denormalization_const = torch.tensor([0.950456, 1.0, 1.088754]) + eps
        self.register_buffer("lab2xyz_xyz_denormalization_const", self.xyz_denormalization_const)

        self.xyz_to_rgb = torch.tensor([
            #     r           g          b
            [ 3.24048134, -0.96925495,  0.05564664], # x
            [-1.53715152,  1.87599, -0.20404134], # y
            [-0.49853633,  0.04155593,  1.05731107], # z
        ]) + eps

        self.register_buffer('xyz2srgb_const', self.xyz_to_rgb)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # with torch.no_grad():
        return self.lab_to_rgb(input)

    def lab_to_rgb(self, input: torch.Tensor) -> torch.Tensor:
        r"""Convert an LAB image to RGB
        The image data is assumed to be in the range of (0, 1).
        Args:
            input (torch.Tensor): LAB Image to be converted to RGB.
        Returns:
            torch.Tensor: RGB version of the image.
        See :class:`~kornia.color.LabToRgb` for details."""
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {type(input)}")

        if not(len(input.shape) == 3 or len(input.shape) == 4):
            raise ValueError(f"Input size must have a shape of (*, 3, H, W) or (3, H, W). Got {input.shape}")

        if input.shape[-3] != 3:
            raise ValueError(f"Expected input to have 3 channels, got {input.shape[-3]}")

        # eps = 10e-12
        # input = input + eps
        # printf("Input to Lab2Rgb: %s\n", (torch.isnan(input) == True).nonzero())

        # printf("input != input: %s\n", (input != input).any())
        # printf("Input to Lab2Rgb: %s\n", input[(torch.isnan(input) == True)])
        # assert ((input != input).any()), "Checkpoint 0"


        input_dims = len(input.shape)
        if input_dims == 3:
            input = input[None, :, :, :]

        # [b, c, h, w] -> [b, h, w, c]
        permuted_input = input.permute(0, 2, 3, 1)
        # self.register_buffer('lab2srgb_permuted_input', permuted_input)
        lab_pixels = permuted_input.reshape(-1, 3)
        # self.register_buffer('lab2srgb_reshaped', lab_pixels)
        xyz_pixels = self.cielab_to_xyz(lab_pixels)
        # self.register_buffer('lab2xyz_xyz', xyz_pixels)
        srgb_pixels = self.xyz_to_srgb(xyz_pixels)
        # self.register_buffer('xyz2srgb_srgb_', srgb_pixels)

        # Reshape to input size
        # [b, h, w, c] -> [b, c, h, w]
        srgb_pixels = srgb_pixels.reshape(permuted_input.size())
        # self.register_buffer('xyz2srgb_srgb_reshaped', srgb_pixels)
        srgb_pixels = srgb_pixels.permute(0, 3, 1, 2)
        # self.register_buffer('xyz2srgb_srgb_permuted', srgb_pixels)

        # assert ((srgb_pixels != srgb_pixels).any()), "Checkpoint 1"

        if input_dims == 3:
            srgb_pixels = srgb_pixels[0, :, :, :]


        # srgb_pixels = srgb_pixels.float()
        # self.register_buffer('lab2srgb_srgb', srgb_pixels)

        # assert((srgb_pixels != srgb_pixels).any()), "Checkpoint 2"
        # printf("srgb_pixels: %s\n", srgb_pixels[(torch.isnan(srgb_pixels) == True)])

        return srgb_pixels

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    def cielab_to_xyz(self, input: torch.Tensor) -> torch.Tensor:
        eps = 10e-12

        lab_to_fxfyfz = self.lab_to_fxfyfz.type(input.data.type())
        fxfyfz_normalization_const = self.fxfyfz_normalization_const.type(input.data.type())
        fxfyfz_pixels = torch.matmul(input + fxfyfz_normalization_const, lab_to_fxfyfz) + eps

        # convert to xyz
        epsilon = 6/29
        linear_mask = (fxfyfz_pixels <= epsilon).type(input.data.type()) + eps
        # self.register_buffer('lab2xyz_linear_mask', linear_mask)

        exponential_mask = (fxfyfz_pixels > epsilon).type(input.data.type()) + eps
        # self.register_buffer('lab2xyz_exponential_mask', exponential_mask)

        xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask
        # self.register_buffer('lab2xyz_xyz_', xyz_pixels)

        # denormalize for D65 white point
        xyz_denormalization_const = self.xyz_denormalization_const.type(input.data.type())
        xyz_pixels = torch.mul(xyz_pixels, xyz_denormalization_const)

        return xyz_pixels


    def xyz_to_srgb(self, input: torch.Tensor) -> torch.Tensor:
        eps = 10e-12
        xyz_to_rgb = self.xyz_to_rgb.type(input.data.type())

        rgb_pixels = torch.matmul(input, xyz_to_rgb)
        # self.register_buffer('xyz2srgb_rgb', rgb_pixels)

        # avoid a slightly negative number messing up the conversion
        rgb_pixels = torch.clamp(rgb_pixels, 0.0, 1.0-eps) + eps
        # self.register_buffer('xyz2srgb_rgb_clamp', rgb_pixels)

        linear_mask = (rgb_pixels <= 0.0031308).type(input.data.type()) + eps
        # self.register_buffer('xyz2srgb_linear_mask', linear_mask)

        exponential_mask = (rgb_pixels > 0.0031308).type(input.data.type()) + eps
        # self.register_buffer('xyz2srgb_exponential_mask', exponential_mask)

        srgb_pixels = (rgb_pixels * 12.92 * linear_mask)
        ##### TODO: problematic grad here!
        # Function 'PowBackward0' returned nan values in its 0th output.

        # rgb_pixels[rgb_pixels.ne(rgb_pixels)] = 0
        # rgb_pixels += 10e-12
        srgb_pixels += ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return srgb_pixels



def rgb_to_lab(input: torch.Tensor) -> torch.Tensor:
    rgbToLabModule = RgbToLab()
    return rgbToLabModule(input)

def lab_to_rgb(input: torch.Tensor) -> torch.Tensor:
    labToRgbModule = LabToRgb()
    return labToRgbModule(input)