from kornia.contrib.models.base import ModelBase, ModelType
from kornia.contrib.sam.architecture.image_encoder import ImageEncoderViT
from kornia.contrib.sam.architecture.mask_decoder import MaskDecoder
from kornia.contrib.sam.architecture.prompt_encoder import PromptEncoder


class PrompterModelBase(ModelBase[ModelType]):
    image_encoder: ImageEncoderViT
    prompt_encoder: PromptEncoder
    mask_decoder: MaskDecoder
