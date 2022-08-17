import importlib
import math
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
    # convert to image array and resize
    img: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)[..., :3]
    # convert the image to a tensor
    img_t: torch.Tensor = K.utils.image_to_tensor(img, keepdim=False)  # 1xCxHXW
    img_t = img_t.float() / 255.0
    if resize_to is None:
        img_t = K.geometry.resize(img_t, 184)
    else:
        img_t = K.geometry.resize(img_t, resize_to)
    return img_t


def main():
    # load the images
    BASE_IMAGE_URL1: str = "https://raw.githubusercontent.com/kornia/data/main/panda.jpg"  # augmentation
    BASE_IMAGE_URL2: str = "https://raw.githubusercontent.com/kornia/data/main/simba.png"  # color
    BASE_IMAGE_URL3: str = "https://raw.githubusercontent.com/kornia/data/main/girona.png"  # enhance
    BASE_IMAGE_URL4: str = "https://raw.githubusercontent.com/kornia/data/main/baby_giraffe.png"  # morphology
    BASE_IMAGE_URL5: str = "https://raw.githubusercontent.com/kornia/data/main/persistencia_memoria.jpg"  # filters
    BASE_IMAGE_URL6: str = "https://raw.githubusercontent.com/kornia/data/main/delorean.png"  # geometry
    OUTPUT_PATH = Path(__file__).absolute().parent / "source/_static/img"

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Pointing images to path {OUTPUT_PATH}.")
    img1 = read_img_from_url(BASE_IMAGE_URL1)
    img2 = read_img_from_url(BASE_IMAGE_URL2, img1.shape[-2:])
    img3 = read_img_from_url(BASE_IMAGE_URL3, img1.shape[-2:])
    img4 = read_img_from_url(BASE_IMAGE_URL4)
    img5 = read_img_from_url(BASE_IMAGE_URL5, (234, 320))
    img6 = read_img_from_url(BASE_IMAGE_URL6)

    # TODO: make this more generic for modules out of kornia.augmentation
    # Dictionary containing the transforms to generate the sample images:
    # Key: Name of the transform class.
    # Value: (parameters, num_samples, seed)
    mod = importlib.import_module("kornia.augmentation")
    augmentations_list: dict = {
        "CenterCrop": ((184, 184), 1, 2018),
        "ColorJiggle": ((0.3, 0.3, 0.3, 0.3), 2, 2018),
        "RandomAffine": (((-15.0, 20.0), (0.1, 0.1), (0.7, 1.3), 20), 2, 2019),
        "RandomBoxBlur": (((7, 7),), 1, 2020),
        "RandomCrop": ((img1.shape[-2:], (50, 50)), 2, 2020),
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
        "RandomPlanckianJitter": ((), 2, 2022),
        "RandomPlasmaShadow": (((0.2, 0.5),), 2, 2022),
        "RandomPlasmaBrightness": ((), 2, 2022),
        "RandomPlasmaContrast": ((), 2, 2022),
        "RandomPosterize": (((1, 4),), 2, 2016),
        "RandomResizedCrop": ((img1.shape[-2:], (1.0, 2.0), (1.0, 2.0)), 2, 2020),
        "RandomRotation": ((45.0,), 2, 2019),
        "RandomSharpness": ((16.0,), 1, 2019),
        "RandomSolarize": ((0.2, 0.2), 2, 2019),
        "RandomVerticalFlip": ((), 1, 2020),
        "RandomThinPlateSpline": ((), 1, 2020),
    }

    # ITERATE OVER THE TRANSFORMS
    for aug_name, (args, num_samples, seed) in augmentations_list.items():
        img_in = img1.repeat(num_samples, 1, 1, 1)
        # dynamically create the class instance
        cls = getattr(mod, aug_name)
        aug = cls(*args, p=1.0)
        # set seed
        torch.manual_seed(seed)
        # apply the augmentation to the image and concat
        out = aug(img_in)

        if aug_name == "CenterCrop":
            h, w = img1.shape[-2:]
            h_new, w_new = out.shape[-2:]
            h_dif, w_dif = int(h - h_new), int(w - w_new)
            out = torch.nn.functional.pad(out, (w_dif // 2, w_dif // 2, 0, h_dif))

        out = torch.cat([img_in[0], *(out[i] for i in range(out.size(0)))], dim=-1)
        # save the output image
        out_np = K.utils.tensor_to_image((out * 255.0).byte())
        cv2.imwrite(str(OUTPUT_PATH / f"{aug_name}.png"), out_np)
        sig = f"{aug_name}({', '.join([str(a) for a in args])}, p=1.0)"
        print(f"Generated image example for {aug_name}. {sig}")

    mod = importlib.import_module("kornia.augmentation")
    mix_augmentations_list: dict = {
        "RandomMixUp": (((0.3, 0.4),), 2, 20),
        "RandomCutMix": ((img1.shape[-2], img1.shape[-1]), 2, 2019),
    }
    # ITERATE OVER THE TRANSFORMS
    for aug_name, (args, num_samples, seed) in mix_augmentations_list.items():
        img_in = torch.cat([img1, img2])
        # dynamically create the class instance
        cls = getattr(mod, aug_name)
        aug = cls(*args, p=1.0)
        # set seed
        torch.manual_seed(seed)
        # apply the augmentation to the image and concat
        out, _ = aug(img_in, torch.tensor([0, 1]))
        out = torch.cat([img_in[0], img_in[1], *(out[i] for i in range(out.size(0)))], dim=-1)
        # save the output image
        out_np = K.utils.tensor_to_image((out * 255.0).byte())
        cv2.imwrite(str(OUTPUT_PATH / f"{aug_name}.png"), out_np)
        sig = f"{aug_name}({', '.join([str(a) for a in args])}, p=1.0)"
        print(f"Generated image example for {aug_name}. {sig}")

    mod = importlib.import_module("kornia.color")
    color_transforms_list: dict = {
        "grayscale_to_rgb": ((), 3),
        "rgb_to_bgr": ((), 1),
        "rgb_to_grayscale": ((), 1),
        "rgb_to_hsv": ((), 1),
        "rgb_to_hls": ((), 1),
        "rgb_to_luv": ((), 1),
        "rgb_to_lab": ((), 1),
        # "rgb_to_rgba": ((1.,), 1),
        "rgb_to_xyz": ((), 1),
        "rgb_to_ycbcr": ((), 1),
        "rgb_to_yuv": ((), 1),
        "rgb_to_linear_rgb": ((), 1),
    }
    # ITERATE OVER THE TRANSFORMS
    for fn_name, (args, num_samples) in color_transforms_list.items():
        # import function and apply
        fn = getattr(mod, fn_name)
        if fn_name == "grayscale_to_rgb":
            out = fn(K.color.rgb_to_grayscale(img2), *args)
        else:
            out = fn(img2, *args)
        # perform normalization to visualize
        if fn_name == "rgb_to_lab":
            out = out[:, :1] / 100.0
        elif fn_name == "rgb_to_hsv":
            out[:, :1] = out[:, :1] / 2 * math.pi
        elif fn_name == "rgb_to_luv":
            out = out[:, :1] / 116.0
        # repeat channels for grayscale
        if out.shape[1] != 3:
            out = out.repeat(1, 3, 1, 1)
        # save the output image
        if fn_name == "grayscale_to_rgb":
            out = torch.cat(
                [K.color.rgb_to_grayscale(img2[0]).repeat(3, 1, 1), *(out[i] for i in range(out.size(0)))], dim=-1
            )
        else:
            out = torch.cat([img2[0], *(out[i] for i in range(out.size(0)))], dim=-1)
        out_np = K.utils.tensor_to_image((out * 255.0).byte())
        cv2.imwrite(str(OUTPUT_PATH / f"{fn_name}.png"), out_np)
        sig = f"{fn_name}({', '.join([str(a) for a in args])})"
        print(f"Generated image example for {fn_name}. {sig}")

    # korna.enhance module
    mod = importlib.import_module("kornia.enhance")
    transforms: dict = {
        "adjust_brightness": ((torch.tensor([0.25, 0.5]),), 2),
        "adjust_contrast": ((torch.tensor([0.65, 0.5]),), 2),
        "adjust_gamma": ((torch.tensor([0.85, 0.75]), 2.0), 2),
        "adjust_hue": ((torch.tensor([-math.pi / 4, math.pi / 4]),), 2),
        "adjust_saturation": ((torch.tensor([1.0, 2.0]),), 2),
        "solarize": ((torch.tensor([0.8, 0.5]), torch.tensor([-0.25, 0.25])), 2),
        "posterize": ((torch.tensor([4, 2]),), 2),
        "sharpness": ((torch.tensor([1.0, 2.5]),), 2),
        "equalize": ((), 1),
        "invert": ((), 1),
        "equalize_clahe": ((), 1),
        "add_weighted": ((0.75, 0.25, 2.0), 1),
    }
    # ITERATE OVER THE TRANSFORMS
    for fn_name, (args, num_samples) in transforms.items():
        img_in = img3.repeat(num_samples, 1, 1, 1)
        if fn_name == "add_weighted":
            args_in = (img_in, args[0], img2, args[1], args[2])
        else:
            args_in = (img_in, *args)
        # import function and apply
        fn = getattr(mod, fn_name)
        out = fn(*args_in)
        # save the output image
        out = torch.cat([img_in[0], *(out[i] for i in range(out.size(0)))], dim=-1)
        out_np = K.utils.tensor_to_image((out * 255.0).byte())
        cv2.imwrite(str(OUTPUT_PATH / f"{fn_name}.png"), out_np)
        sig = f"{fn_name}({', '.join([str(a) for a in args])})"
        print(f"Generated image example for {fn_name}. {sig}")

    # korna.morphology module
    mod = importlib.import_module("kornia.morphology")
    kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    transforms: dict = {
        "dilation": ((kernel,), 1),
        "erosion": ((kernel,), 1),
        "opening": ((kernel,), 1),
        "closing": ((kernel,), 1),
        "gradient": ((kernel,), 1),
        "top_hat": ((kernel,), 1),
        "bottom_hat": ((kernel,), 1),
    }
    # ITERATE OVER THE TRANSFORMS
    for fn_name, (args, num_samples) in transforms.items():
        img_in = img4.repeat(num_samples, 1, 1, 1)
        args_in = (img_in, *args)
        # import function and apply
        # import pdb;pdb.set_trace()
        fn = getattr(mod, fn_name)
        out = fn(*args_in)
        # save the output image
        out = torch.cat([img_in[0], *(out[i] for i in range(out.size(0)))], dim=-1)
        out_np = K.utils.tensor_to_image((out * 255.0).byte())
        cv2.imwrite(str(OUTPUT_PATH / f"{fn_name}.png"), out_np)
        sig = f"{fn_name}({', '.join([str(a) for a in args])})"
        print(f"Generated image example for {fn_name}. {sig}")

    # korna.filters module
    mod = importlib.import_module("kornia.filters")
    kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    transforms: dict = {
        "box_blur": (((5, 5),), 1),
        "median_blur": (((5, 5),), 1),
        "gaussian_blur2d": (((5, 5), (1.5, 1.5)), 1),
        "motion_blur": ((5, 90.0, 1.0), 1),
        "max_blur_pool2d": ((5,), 1),
        "blur_pool2d": ((5,), 1),
        "unsharp_mask": (((5, 5), (1.5, 1.5)), 1),
        "laplacian": ((5,), 1),
        "sobel": ((), 1),
        "spatial_gradient": ((), 1),
        "canny": ((), 1),
    }
    # ITERATE OVER THE TRANSFORMS
    for fn_name, (args, num_samples) in transforms.items():
        img_in = img5.repeat(num_samples, 1, 1, 1)
        args_in = (img_in, *args)
        # import function and apply
        fn = getattr(mod, fn_name)
        out = fn(*args_in)
        if fn_name in ("max_blur_pool2d", "blur_pool2d"):
            out = K.geometry.resize(out, img_in.shape[-2:])
        if fn_name == "canny":
            out = out[1].repeat(1, 3, 1, 1)
        if isinstance(out, torch.Tensor):
            out = out.clamp(min=0.0, max=1.0)
        if fn_name in ("laplacian", "sobel", "spatial_gradient", "canny"):
            out = K.enhance.normalize_min_max(out)
        if fn_name == "spatial_gradient":
            out = out.permute(2, 1, 0, 3, 4).squeeze()
        # save the output image
        out = torch.cat([img_in[0], *(out[i] for i in range(out.size(0)))], dim=-1)
        out_np = K.utils.tensor_to_image((out * 255.0).byte())
        cv2.imwrite(str(OUTPUT_PATH / f"{fn_name}.png"), out_np)
        sig = f"{fn_name}({', '.join([str(a) for a in args])})"
        print(f"Generated image example for {fn_name}. {sig}")

    # korna.geometry.transform module
    mod = importlib.import_module("kornia.geometry.transform")
    h, w = img6.shape[-2:]

    def _get_tps_args():
        src = torch.tensor([[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [0.0, 0.0]]]).repeat(2, 1, 1)  # Bx5x2
        dst = src + torch.distributions.Uniform(-0.2, 0.2).rsample((2, 5, 2))
        kernel, affine = K.geometry.transform.get_tps_transform(dst, src)
        return src, kernel, affine

    transforms: dict = {
        "warp_affine": (
            (
                K.geometry.transform.get_affine_matrix2d(
                    translations=torch.zeros(2, 2),
                    center=(torch.tensor([w, h]) / 2).repeat(2, 1),
                    scale=torch.distributions.Uniform(0.5, 1.5).rsample((2, 2)),
                    angle=torch.tensor([-25.0, 25.0]),
                )[:, :2, :3],
                (h, w),
            ),
            2,
        ),
        "remap": (
            (
                *(K.utils.create_meshgrid(h, w, normalized_coordinates=True) - 0.25).unbind(-1),
                'bilinear',
                'zeros',
                True,
                True,
            ),
            1,
        ),
        "warp_image_tps": ((_get_tps_args()), 2),
        "rotate": ((torch.tensor([-15.0, 25.0]),), 2),
        "translate": ((torch.tensor([[10.0, -15], [50.0, -25.0]]),), 2),
        "scale": ((torch.tensor([[0.5, 1.25], [1.0, 1.5]]),), 2),
        "shear": ((torch.tensor([[0.1, -0.2], [-0.2, 0.1]]),), 2),
        "rot180": ((), 1),
        "hflip": ((), 1),
        "vflip": ((), 1),
        "resize": (((120, 220),), 1),
        "rescale": ((0.5,), 1),
        "elastic_transform2d": ((torch.rand(1, 2, h, w) * 2 - 1, (63, 63), (32, 32), (4.0, 4.0)), 1),
        "pyrdown": ((), 1),
        "pyrup": ((), 1),
        "build_pyramid": ((3,), 1),
        "build_laplacian_pyramid": ((3,), 1),
    }
    # ITERATE OVER THE TRANSFORMS
    for fn_name, (args, num_samples) in transforms.items():
        img_in = img6.repeat(num_samples, 1, 1, 1)
        args_in = (img_in, *args)
        # import function and apply
        fn = getattr(mod, fn_name)
        out = fn(*args_in)
        if fn_name in ("resize", "rescale", "pyrdown", "pyrup"):
            h_new, w_new = out.shape[-2:]
            out = torch.nn.functional.pad(out, (0, (w - w_new), 0, (h - h_new)))
        if fn_name == "build_pyramid":
            _out = []
            for pyr in out[1:]:
                h_new, w_new = pyr.shape[-2:]
                out_tmp = torch.nn.functional.pad(pyr, (0, (w - w_new), 0, (h - h_new)))
                _out.append(out_tmp)
            out = torch.cat(_out)

        if fn_name == "build_laplacian_pyramid":
            h_, w_ = out[0].shape[-2:]
            _out = [out[0]]
            for pyr in out[1:]:
                h_new, w_new = pyr.shape[-2:]
                out_tmp = torch.nn.functional.pad(pyr, (0, (w_ - w_new), 0, (h_ - h_new)))
                print(out_tmp.size())
                _out.append(out_tmp)
            out = torch.cat(_out)

        # save the output image
        if fn_name != "build_laplacian_pyramid":
            out = torch.cat([img_in[0], *(out[i] for i in range(out.size(0)))], dim=-1)
        else:
            out = torch.cat([*(out[i] for i in range(out.size(0)))], dim=-1)
        out_np = K.utils.tensor_to_image((out * 255.0).byte())
        cv2.imwrite(str(OUTPUT_PATH / f"{fn_name}.png"), out_np)
        sig = f"{fn_name}({', '.join([str(a) for a in args])})"
        print(f"Generated image example for {fn_name}. {sig}")


if __name__ == "__main__":
    main()
