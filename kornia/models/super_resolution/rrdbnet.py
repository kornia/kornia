import torch
from torch import nn

from kornia.config import kornia_config
from kornia.core.external import basicsr
from kornia.models.utils import OutputRangePostProcessor
from kornia.utils.download import CachedDownloader

from .base import SuperResolution

__all__ = ["RRDBNetBuilder"]

URLs = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}


class RRDBNetBuilder:
    @staticmethod
    def build(model_name: str = "RealESRNet_x4plus", pretrained: bool = True) -> SuperResolution:
        if model_name == "RealESRGAN_x4plus":
            model = basicsr.archs.rrdbnet_arch.RRDBNet(  # type: ignore
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
        elif model_name == "RealESRNet_x4plus":
            model = basicsr.archs.rrdbnet_arch.RRDBNet(  # type: ignore
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
        elif model_name == "RealESRGAN_x4plus_anime_6B":
            model = basicsr.archs.rrdbnet_arch.RRDBNet(  # type: ignore
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
            )
        elif model_name == "RealESRGAN_x2plus":
            model = basicsr.archs.rrdbnet_arch.RRDBNet(  # type: ignore
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
            )
        else:
            raise ValueError(
                f"Model {model_name} not found. Please choose from "
                "'RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus'."
            )

        model_path = None
        if pretrained:
            url = URLs[model_name]
            model_path = CachedDownloader.download_to_cache(
                url, model_name, download=True, suffix=".pth", cache_dir=kornia_config.hub_onnx_dir
            )
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))["params_ema"], strict=True)
        model.eval()

        return SuperResolution(
            model,
            pre_processor=nn.Identity(),
            post_processor=OutputRangePostProcessor(min_val=0.0, max_val=1.0),
            name=model_name,
        )
