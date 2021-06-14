import importlib
import requests
from pathlib import Path

import torch
import kornia as K

import cv2
import numpy as np

mod = importlib.import_module("kornia.augmentation")

BASE_IMAGE_URL: str = "https://raw.githubusercontent.com/kornia/data/main/panda.jpg"
OUTPUT_PATH = Path(__file__).absolute().parent / "source/_static"

# perform request
response =  requests.get(BASE_IMAGE_URL).content

# convert to array of ints
nparr = np.frombuffer(response, np.uint8)

# convert to image array
img: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

# convert the image to a tensor
img_t: torch.Tensor = K.utils.image_to_tensor(img, keepdim=False)  # 1xCxHXW
img_t = img_t.float() / 255.

# TODO: make this more generic for modules out of kornia.augmentation
# Dictionary containing the transforms to generate the sample images:
# Key: Name of the transform class.
# Value: The default parameters to use.
augmentations_list: dict = {
    "ColorJitter": (0.1, 0.1, 0.1, 0.1),
    "RandomAffine": ((-15., 20.),),
    "RandomBoxBlur": ((7, 7),),
    "RandomChannelShuffle": (),
    "RandomErasing": ((.4, .8), (.3, 1/.3),),
    "RandomElasticTransform": ((63, 63), (32, 32), (1., 1.,),),
    "RandomEqualize": (),
    "RandomFisheye": (torch.tensor([-.3, .3]), torch.tensor([-.3, .3]), torch.tensor([.9, 1.]),),
    "RandomGrayscale": (),
    "RandomGaussianNoise": (0., .05),
    "RandomHorizontalFlip": (),
    "RandomInvert": (),
    "RandomMotionBlur": (7, 35., 0.5,),
    "RandomPerspective": (0.2,),
    "RandomPosterize": (3,),
    "RandomResizedCrop": ((510, 1020), (3., 3.), (2., 2.)),
    "RandomRotation": (45.,),
    "RandomSharpness": (1.,),
    "RandomSolarize": (0.1,),
    "RandomVerticalFlip": (),
    "RandomThinPlateSpline": (),
}

# ITERATE OVER THE TRANSFORMS

for aug_name, args in augmentations_list.items():
    # dynamically create the class instance
    cls = getattr(mod, aug_name)
    aug = cls(*args, p=1.0)
    # apply the augmentaiton to the image and concat
    out = aug(img_t)
    out = torch.cat([img_t, out], dim=-1)
    # save the output image
    out_np = K.utils.tensor_to_image((out * 255.).byte())
    cv2.imwrite(str(OUTPUT_PATH / f"{aug_name}.jpg"), out_np)
