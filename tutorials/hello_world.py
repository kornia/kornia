"""

Hello world: Planet Kornia
==========================

Welcome to Planet Kornia: a set of tutorial to learn about Computer Vision in PyTorch.

This is the first tutorial showing how one can simply load an image and convert from BGR to RGB using Kornia.

"""

import torch
import kornia
import cv2
import numpy as np

import matplotlib.pyplot as plt

#############################
# We use OpenCV to load an image to memory represented in a numpy.ndarray
img_bgr: np.ndarray = cv2.imread('./data/arturito.jpeg')  # HxWxC

#############################
# The image is convert to a 4D torch tensor
x_bgr: torch.tensor = kornia.image_to_tensor(img_bgr)  # 1xCxHxW

#############################
# Once with a torch tensor we can use any Kornia operator
x_rgb: torch.tensor = kornia.bgr_to_rgb(x_bgr)  # 1xCxHxW

#############################
# Convert back to numpy to visualize
img_rgb: np.ndarray = kornia.tensor_to_image(x_rgb.byte())  # HxWxC

#############################
# We use Matplotlib to visualize de results
fig, axs = plt.subplots(1, 2, figsize=(32, 16))
axs = axs.ravel()

axs[0].axis('off')
axs[0].imshow(img_bgr)

axs[1].axis('off')
axs[1].imshow(img_rgb)
