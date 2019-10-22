import torch
import kornia
import cv2
import numpy as np 
import pdb #todo remove
import matplotlib.pyplot as plt
 

#read the image with Opencv
img: np.array = cv2.imread('./data/doraemon.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
th,img = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY)

# convert to torch tensor
bin_image: torch.tensor = kornia.image_to_tensor(img).float()

# structuring_element is a torch.tensor of ndims 2 containing only 1's and 0's
structuring_element = torch.tensor(np.ones([5,5])).float()

dilated_image = kornia.morphology.dilation(bin_image, structuring_element)

# convert back to numpy
dilated_image: np.array = kornia.tensor_to_image(dilated_image)

pdb.set_trace()

# Create the plot
fig, axs = plt.subplots(1, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('Original image')
axs[0].imshow(img, cmap='gray', vmin=0, vmax=1.0)

axs[1].axis('off')
axs[1].set_title('Dilated image')
axs[1].imshow(dilated_image, cmap='gray', vmin=0, vmax=1.0)

plt.show()
