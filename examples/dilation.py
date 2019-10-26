import torch
import kornia
import cv2
import numpy as np
import pdb #todo remove
import matplotlib.pyplot as plt


# create an image
img = np.zeros([1,10,10], dtype=float)
img[:,3:6,3:6] = 1.0
img[:,3,3] = 0.0
img[:,4:5,4:5] = 0.0
img[:,6:8,6] = 1.0

# convert to torch tensor
bin_image: torch.tensor = torch.tensor(img, dtype=torch.float32)

# structuring_element is a torch.tensor of ndims 2 containing only 1's and 0's
structuring_element = torch.tensor(np.ones([3,3])).float()

dilated_image = kornia.morphology.dilation(bin_image, structuring_element)

# convert back to numpy
dilated_image: np.array = kornia.tensor_to_image(dilated_image)

# Create the plot
fig, axs = plt.subplots(1, 3, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('Original image')
axs[0].imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1.0)

axs[1].axis('off')
axs[1].set_title('Dilated image')
axs[1].imshow(dilated_image, cmap='gray', vmin=0, vmax=1.0)

axs[2].axis('off')
axs[2].set_title('Superimposed')
axs[2].imshow(img.squeeze()*0.5 + dilated_image * 0.5, cmap='gray', vmin=0, vmax=1.0)

plt.show()
