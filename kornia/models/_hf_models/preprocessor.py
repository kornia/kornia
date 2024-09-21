from typing import Any

import requests

from kornia.core import ImageSequential, Tensor, tensor
from kornia.enhance.normalize import Normalize
from kornia.enhance.rescale import Rescale
from kornia.geometry.transform import Resize


class PreprocessingLoader:
    @staticmethod
    def normalize(mean: Tensor, std: Tensor) -> Normalize:
        return Normalize(mean=mean, std=std)

    @staticmethod
    def rescale(rescale_factor: float) -> Rescale:
        return Rescale(factor=rescale_factor)

    @staticmethod
    def resize(width: int, height: int) -> Resize:
        return Resize((height, width))

    @staticmethod
    def get_json_from_url(url: str) -> dict[str, Any]:
        req = requests.get(url, headers={"Accept": "application/json"})
        return req.json()

    @staticmethod
    def from_url(url: str) -> "PreprocessingLoader":
        req = PreprocessingLoader.get_json_from_url(url)
        if req["image_processor_type"] == "DPTImageProcessor":
            return DPTImageProcessor.from_json(req)
        raise RuntimeError(f"Unsupported image processor type: {req['image_processor_type']}")

    @staticmethod
    def from_json(req: dict[str, Any]) -> "PreprocessingLoader":
        if req["image_processor_type"] == "DPTImageProcessor":
            return DPTImageProcessor.from_json(req)
        raise RuntimeError(f"Unsupported image processor type: {req['image_processor_type']}")


class DPTImageProcessor(PreprocessingLoader):
    @staticmethod
    def from_json(json_data: dict[str, Any]) -> ImageSequential:
        preproc_list = []
        if json_data["do_pad"]:
            raise NotImplementedError
        if json_data["do_resize"]:
            # Missing some parameters such as `ensure_multiple_of`, `keep_aspect_ratio`
            preproc_list.append(
                PreprocessingLoader.resize(width=json_data["size"]["width"], height=json_data["size"]["height"])
            )
        if json_data["do_rescale"]:
            preproc_list.append(PreprocessingLoader.rescale(rescale_factor=json_data["rescale_factor"] * 255))
        if json_data["do_normalize"]:
            preproc_list.append(
                PreprocessingLoader.normalize(
                    mean=tensor([json_data["image_mean"]]) / 255, std=tensor([json_data["image_std"]]) / 255
                )
            )
        return ImageSequential(*preproc_list)
