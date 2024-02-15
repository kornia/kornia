# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

# TODO: promote this to kornia
from kornia.contrib.models.efficient_vit.utils.list import val2tuple
from kornia.contrib.models.efficient_vit.utils.network import build_kwargs_from_config, get_same_padding

__all__ = ["val2tuple", "get_same_padding", "build_kwargs_from_config"]
