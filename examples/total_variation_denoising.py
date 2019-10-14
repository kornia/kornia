"""

Denoise image using total variation
======================================

"""

import torch
import kornia
import cv2
import numpy as np

import matplotlib.pyplot as plt

# read the image with OpenCV
img: np.array = cv2.imread('./data/doraemon.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
img = img + np.random.normal(loc=0.0,scale=0.1,size=img.shape)
img = np.clip(img, 0.0, 1.0)

# convert to torch tensor
noisy_image: torch.tensor = kornia.image_to_tensor(img).squeeze()  # CxHxW


# define the total variation denoising network
class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction='mean')
        self.regularization_term = kornia.losses.TotalVariation()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image

    def forward(self):
        return self.l2_term(self.clean_image, self.noisy_image) + 0.0001 * self.regularization_term(self.clean_image)

    def get_clean_image(self):
        return self.clean_image

tv_denoiser = TVDenoise(noisy_image)

# define the optimizer to optimize the 1 parameter of tv_denoiser
optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr=0.1, momentum=0.9)

# run the optimization loop
num_iters = 500
for i in range(num_iters):
    optimizer.zero_grad()
    loss = tv_denoiser()
    if i%25 == 0:
        print("Loss in iteration ", i, " of ", num_iters, ": ", loss.item())
    loss.backward()
    optimizer.step()

# convert back to numpy
img_clean: np.array = kornia.tensor_to_image(tv_denoiser.get_clean_image())

# Create the plot
fig, axs = plt.subplots(1, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('Noisy image')
axs[0].imshow(img)

axs[1].axis('off')
axs[1].set_title('Cleaned image')
axs[1].imshow(img_clean)

plt.show()
