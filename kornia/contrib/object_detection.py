from __future__ import annotations

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor, concatenate


class ObjectDetector(Module):
    def __init__(self, model: Module, eval_size: int | tuple[int, int], interpolation_mode: str = "bilinear") -> None:
        super().__init__()
        self.model = model
        self.eval_size = eval_size
        self.interpolation_mode = interpolation_mode

    def forward(self, imgs: list[Tensor]) -> list[Tensor]:
        """Detect objects in a given list of images.

        Args:
            imgs: list of RGB images. Each image has shape :math:`(3, H, W)`.

        Returns:
            list of detections found in each image. For item in a batch, shape is :math:`(D, 6)`, where :math:`D` is the
            number of detections in the given image, :math:`6` represents class id, score, and `xywh` bounding box.
        """
        # NOTE: also supports other input format? e.g. PIL Image
        original_sizes = [(img.shape[1], img.shape[2]) for img in imgs]
        resized_imgs = [F.interpolate(img.unsqueeze(0), self.eval_size, mode=self.interpolation_mode) for img in imgs]
        return self.model(concatenate(resized_imgs), original_sizes)

    def compile(
        self,
        *,
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = 'inductor',
        mode: str | None = None,
        options: dict[str, str | int | bool] | None = None,
        disable: bool = False,
    ) -> None:
        self.model = torch.compile(  # type: ignore
            self.model,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )
