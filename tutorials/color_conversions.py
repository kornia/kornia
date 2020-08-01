"""
Color space conversions
=======================

In this tutorial we are going to learn how to convert image from different image spaces using `kornia.color`.

"""

from matplotlib import pyplot as plt
import cv2
import numpy as np

import torch
import kornia
import torchvision

#############################
# We use OpenCV to load an image to memory represented in a numpy.ndarray
img_bgr: np.ndarray = cv2.imread('./data/simba.png', cv2.IMREAD_COLOR)

#############################
# Convert the numpy array to torch
x_bgr: torch.Tensor = kornia.image_to_tensor(img_bgr, keepdim=False)

#############################
# Using `kornia` we easily perform color transformation in batch mode.


def hflip(input: torch.Tensor) -> torch.Tensor:
    return torch.flip(input, [-1])


def vflip(input: torch.Tensor) -> torch.Tensor:
    return torch.flip(input, [-2])


def rot180(input: torch.Tensor) -> torch.Tensor:
    return torch.flip(input, [-2, -1])


def imshow(input: torch.Tensor):
    out: torch.Tensor = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = kornia.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()

#############################
# Create a batch of images
xb_bgr = torch.cat([x_bgr, hflip(x_bgr), vflip(x_bgr), rot180(x_bgr)])
imshow(xb_bgr)

#############################
# Convert BGR to RGB
xb_rgb = kornia.bgr_to_rgb(xb_bgr)
imshow(xb_rgb)

#############################
# Convert RGB to grayscale
# NOTE: image comes in torch.uint8, and kornia assumes floating point type
xb_gray = kornia.rgb_to_grayscale(xb_rgb.float() / 255.)
imshow(xb_gray)

#############################
# Convert RGB to HSV
xb_hsv = kornia.rgb_to_hsv(xb_rgb.float() / 255.)
imshow(xb_hsv[:, 2:3])

#############################
# Convert RGB to YUV
# NOTE: image comes in torch.uint8, and kornia assumes floating point type
yuv = kornia.rgb_to_yuv(xb_rgb.float() / 255.)
y_channel = torchvision.utils.make_grid(yuv, nrow=2)[0, :, :]
plt.imshow(y_channel, cmap='gray', vmin=0, vmax=1)  # Displaying only y channel
plt.axis('off')
plt.show()
