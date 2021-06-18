import importlib
import os
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

import kornia as K


def main():

    mod = importlib.import_module("kornia.augmentation")

    BASE_IMAGE_URL: str = "https://raw.githubusercontent.com/kornia/data/main/panda.jpg"
    OUTPUT_PATH = Path(__file__).absolute().parent / "source/_static/img"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Pointing images to path {OUTPUT_PATH}.")

    # perform request
    response = requests.get(BASE_IMAGE_URL).content

    # convert to array of ints
    nparr = np.frombuffer(response, np.uint8)

    # convert to image array
    img: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (186, int(img.shape[0] / img.shape[1] * 224)))

    # convert the image to a tensor
    img_t: torch.Tensor = K.utils.image_to_tensor(img, keepdim=False)  # 1xCxHXW
    img_t = img_t.float() / 255.0

    # TODO: make this more generic for modules out of kornia.augmentation
    # Dictionary containing the transforms to generate the sample images:
    # Key: Name of the transform class.
    # Value: (parameters, num_samples, seed)
    augmentations_list: dict = {
        "ColorJitter": ((0.3, 0.3, 0.3, 0.3), 2, 2018),
        "RandomAffine": (((-15.0, 20.0), (0.1, 0.1), (0.7, 1.3), 20), 2, 2019),
        "RandomBoxBlur": (((7, 7),), 1, 2020),
        "RandomChannelShuffle": ((), 1, 2020),
        "RandomElasticTransform": (((63, 63), (32, 32), (2.0, 2.0)), 2, 2018),
        "RandomEqualize": ((), 1, 2020),
        "RandomErasing": (((0.2, 0.4), (0.3, 1 / 0.3)), 2, 2017),
        "RandomFisheye": ((torch.tensor([-0.3, 0.3]), torch.tensor([-0.3, 0.3]), torch.tensor([0.9, 1.0])), 2, 2020),
        "RandomGaussianBlur": (((3, 3), (0.1, 2.0)), 1, 2020),
        "RandomGaussianNoise": ((0.0, 0.05), 1, 2020),
        "RandomGrayscale": ((), 1, 2020),
        "RandomHorizontalFlip": ((), 1, 2020),
        "RandomInvert": ((), 1, 2020),
        "RandomMotionBlur": ((7, 35.0, 0.5), 2, 2020),
        "RandomPerspective": ((0.2,), 2, 2020),
        "RandomPosterize": (((1, 4),), 2, 2016),
        "RandomResizedCrop": ((img.shape[:2], (1.0, 2.0), (1.0, 2.0)), 2, 2020),
        "RandomRotation": ((45.0,), 2, 2019),
        "RandomSharpness": ((16.0,), 1, 2019),
        "RandomSolarize": ((0.2, 0.2,), 2, 2019),
        "RandomVerticalFlip": ((), 1, 2020),
        "RandomThinPlateSpline": ((), 1, 2020),
    }

    # ITERATE OVER THE TRANSFORMS

    for aug_name, (args, num_samples, seed) in augmentations_list.items():
        img_in = img_t.repeat(num_samples, 1, 1, 1)
        # dynamically create the class instance
        cls = getattr(mod, aug_name)
        aug = cls(*args, p=1.0)
        # set seed
        torch.manual_seed(seed)
        # apply the augmentaiton to the image and concat
        out = aug(img_in)
        out = torch.cat([img_in[0], *[out[i] for i in range(out.size(0))]], dim=-1)
        # save the output image
        out_np = K.utils.tensor_to_image((out * 255.0).byte())
        cv2.imwrite(str(OUTPUT_PATH / f"{aug_name}.png"), out_np)
        print(f"Generated image example for {aug_name}.")


if __name__ == "__main__":
    main()
