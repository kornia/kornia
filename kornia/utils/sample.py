import logging
import os
from typing import Any, List, Optional, Tuple, Union

import kornia
from kornia.core import Tensor, stack
from kornia.core.external import PILImage as Image
from kornia.core.external import requests
from kornia.io import load_image

__all__ = [
    "get_sample_images",
]

IMAGE_URLS: List[str] = [
    "https://raw.githubusercontent.com/kornia/data/main/panda.jpg",
    "https://raw.githubusercontent.com/kornia/data/main/simba.png",
    "https://raw.githubusercontent.com/kornia/data/main/girona.png",
    "https://raw.githubusercontent.com/kornia/data/main/baby_giraffe.png",
    "https://raw.githubusercontent.com/kornia/data/main/persistencia_memoria.jpg",
    "https://raw.githubusercontent.com/kornia/data/main/delorean.png",
]


def download_image(url: str, save_to: str) -> None:
    """Download an image from a given URL and save it to a specified file path.

    Args:
        url: The URL of the image to download.
        save_to: The file path where the downloaded image will be saved.
    """
    im = Image.open(requests.get(url, stream=True, timeout=30).raw)  # type:ignore
    im.save(save_to)


def get_sample_images(
    resize: Optional[Tuple[int, int]] = None,
    paths: List[str] = IMAGE_URLS,
    download: bool = True,
    cache_dir: Optional[str] = None,
    as_list: Optional[bool] = None,
    divisible_factor: Optional[int] = None,
    **kwargs: Any,
) -> Union[Tensor, List[Tensor]]:
    """Loads multiple images from the given URLs.

    Optionally download them, resize them if specified, and return them as a batch of tensors or a list of tensors.

    Args:
        paths: A list of path or URL from which to load or download images.
              Defaults to a pre-defined constant `IMAGE_URLS` if not provided.
        resize: Optional target size for resizing all images as a tuple (height, width).
            If not provided, the images will not be resized, and their original sizes will be retained.
        download: Whether to download the images if they are not already cached. Defaults to True.
        cache_dir: The directory where the downloaded images will be cached.
            Defaults to ".kornia_hub/images".
        as_list: if to keep the output as a list. If None and `resize` is None, the output will be a list.
            If None and `resize` is not None, the output will be a single tensor.
        divisible_factor:
            If not None, the images will be resized to the nearest multiple of `divisible_factor`.
        **kwargs: Additional keyword arguments to pass to `kornia.geometry.resize`.

    Returns:
        torch.Tensor | list[torch.Tensor]:
            If `resize` is provided, returns a single stacked tensor with shape (B, C, H, W).
            Otherwise, returns a list of tensors, each with its original shape (C, H, W).
    """
    if cache_dir is None:
        cache_dir = ".kornia_hub/images"
    os.makedirs(cache_dir, exist_ok=True)
    tensors = []
    for path in paths:
        if path.startswith("http"):
            name = os.path.basename(path)
            fname = os.path.join(cache_dir, name)
            if not os.path.exists(fname) and download:
                logging.info(f"Downloading `{path}` to `{fname}`.")
                download_image(path, fname)
            elif not os.path.exists(fname) and not download:
                logging.error(
                    f"Image `{path}` not found at `{fname}`. You may want to set `download=True` to download it."
                )
        else:
            fname = path
        img_tensor = load_image(fname)
        if resize is not None:
            img_tensor = kornia.geometry.resize(img_tensor, resize, **kwargs)
        if divisible_factor is not None:
            img_tensor = kornia.geometry.transform.resize_to_be_divisible(img_tensor, divisible_factor, **kwargs)
        tensors.append(img_tensor)

    if not as_list and resize is not None:
        return stack(tensors)

    return tensors
