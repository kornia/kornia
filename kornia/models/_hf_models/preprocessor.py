from future import __annotations__

import requests
from typing import Any

from kornia.core import ImageSequential
from kornia.enhance.normalize import Normalize
from kornia.enhance.rescale import Rescale
from kornia.geometry.transform import Resize


class PreprocessingLoader:
    
    @staticmethod
    def normalize(mean: list[float], std: list[float]) -> Normalize:
        return Normalize(mean=mean, std=std)

    @staticmethod
    def rescale(rescale_factor: float)-> Rescale:
        return Rescale(factor=rescale_factor)

    @staticmethod
    def resize(width: int, height: int) -> Resize:
        return Resize((height, width))

    @staticmethod
    def from_url(url: str) -> "PreprocessingLoader":
        req = requests.get(url, headers={'Accept': 'application/json'})
        if req["image_processor_type"] == "DPTImageProcessor":
            return DPTImageProcessor.from_json(req)
        raise RuntimeError(f"Unsupported image processor type: {req['image_processor_type']}")


class DPTImageProcessor(PreprocessingLoader):

    @staticmethod
    def from_json(json: dict[str, Any]) -> ImageSequential:
        preproc_list = [] 
        if json["do_pad"]:
            raise NotImplementedError
        if json["do_resize"]:
            # Missing some parameters such as `ensure_multiple_of`, `keep_aspect_ratio`
            preproc_list.append(PreprocessingLoader.resize(width=json["size"]["width"], height=json["size"]["height"]))
        if json["do_rescale"]:
            preproc_list.append(PreprocessingLoader.rescale(rescale_factor=json["rescale_factor"] * 255))
        if json["do_normalize"]:
            preproc_list.append(PreprocessingLoader.normalize(mean=json["mean"] / 255, std=json["std"] / 255))
        return ImageSequential(*preproc_list)
