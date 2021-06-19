import importlib
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import torch

import kornia as K


def read_img_from_url(url: str, resize_to: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    # perform request
    response = requests.get(url).content
    # convert to array of ints
    nparr = np.frombuffer(response, np.uint8)
    # convert to image array
    img: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if resize_to is None:
        img = cv2.resize(img, (186, int(img.shape[0] / img.shape[1] * 224)))
    else:
        img = cv2.resize(img, resize_to)
    # convert the image to a tensor
    img_t: torch.Tensor = K.utils.image_to_tensor(img, keepdim=False)  # 1xCxHXW
    img_t = img_t.float() / 255.0
    return img_t


def main():

    mod = importlib.import_module("kornia.augmentation")

    BASE_IMAGE_URL: str = "https://raw.githubusercontent.com/kornia/data/main/panda.jpg"
    BASE_IMAGE_URL2: str = "https://raw.githubusercontent.com/kornia/kornia/master/tutorials/data/simba.png"
    OUTPUT_PATH = Path(__file__).absolute().parent / "source/_static/img"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Pointing images to path {OUTPUT_PATH}.")
    img = read_img_from_url(BASE_IMAGE_URL)
    img2 = read_img_from_url(BASE_IMAGE_URL2, img.shape[-2:][::-1])

    # TODO: make this more generic for modules out of kornia.augmentation
    # Dictionary containing the transforms to generate the sample images:
    # Key: Name of the transform class.
    # Value: (parameters, num_samples, seed)
    augmentations_list: dict = {
        "ColorJitter": ((0.3, 0.3, 0.3, 0.3), 2, 2018),
        "RandomAffine": (((-15.0, 20.0), (0.1, 0.1), (0.7, 1.3), 20), 2, 2019),
        "RandomBoxBlur": (((7, 7),), 1, 2020),
        "RandomCrop": ((img.shape[-2:], (50, 50)), 2, 2020),
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
        "RandomResizedCrop": ((img.shape[-2:], (1.0, 2.0), (1.0, 2.0)), 2, 2020),
        "RandomRotation": ((45.0,), 2, 2019),
        "RandomSharpness": ((16.0,), 1, 2019),
        "RandomSolarize": ((0.2, 0.2,), 2, 2019),
        "RandomVerticalFlip": ((), 1, 2020),
        "RandomThinPlateSpline": ((), 1, 2020),
    }

    # ITERATE OVER THE TRANSFORMS
    # for aug_name, (args, num_samples, seed) in augmentations_list.items():
    #     img_in = img.repeat(num_samples, 1, 1, 1)
    #     # dynamically create the class instance
    #     cls = getattr(mod, aug_name)
    #     aug = cls(*args, p=1.0)
    #     # set seed
    #     torch.manual_seed(seed)
    #     # apply the augmentaiton to the image and concat
    #     out = aug(img_in)
    #     out = torch.cat([img_in[0], *[out[i] for i in range(out.size(0))]], dim=-1)
    #     # save the output image
    #     out_np = K.utils.tensor_to_image((out * 255.0).byte())
    #     cv2.imwrite(str(OUTPUT_PATH / f"{aug_name}.png"), out_np)
    #     sig = f"{aug_name}({', '.join([str(a) for a in args])}, p=1.0)"
    #     print(f"Generated image example for {aug_name}. {sig}")

    mix_augmentations_list: dict = {
        "RandomMixUp": (((.3, .4),), 2, 20),
        "RandomCutMix": ((img.shape[-2], img.shape[-1]), 2, 2019),
    }
    # ITERATE OVER THE TRANSFORMS
    for aug_name, (args, num_samples, seed) in mix_augmentations_list.items():
        img_in = torch.cat([img, img2])
        # dynamically create the class instance
        cls = getattr(mod, aug_name)
        aug = cls(*args, p=1.0)
        # set seed
        torch.manual_seed(seed)
        # apply the augmentaiton to the image and concat
        out, _ = aug(img_in, torch.tensor([0, 1]))
        out = torch.cat([img_in[0], img_in[1], *[out[i] for i in range(out.size(0))]], dim=-1)
        # save the output image
        out_np = K.utils.tensor_to_image((out * 255.0).byte())
        cv2.imwrite(str(OUTPUT_PATH / f"{aug_name}.png"), out_np)
        sig = f"{aug_name}({', '.join([str(a) for a in args])}, p=1.0)"
        print(f"Generated image example for {aug_name}. {sig}")


if __name__ == "__main__":
    main()
