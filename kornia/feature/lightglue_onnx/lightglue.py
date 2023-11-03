from __future__ import annotations

from typing import ClassVar

import torch

from kornia.core import Device, Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_DEVICES, KORNIA_CHECK_SHAPE

from .utils import download_onnx_from_url, normalize_keypoints

try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    np = None  # type: ignore
    ort = None

__all__ = ["OnnxLightGlue"]


class OnnxLightGlue:
    r"""Wrapper for loading LightGlue-ONNX models and running inference via ONNXRuntime.

    LightGlue :cite:`LightGlue2023` performs fast descriptor-based deep keypoint matching.
    This module requires `onnxruntime` to be installed.

    If you have trained your own LightGlue model, see https://github.com/fabio-sim/LightGlue-ONNX
    for how to export the model to ONNX and optimize it.

    Args:
        weights: Pretrained weights, or a path to your own exported ONNX model. Available pretrained weights
          are ``'disk'``, ``'superpoint'``, ``'disk_fp16'``, and ``'superpoint_fp16'``. `Note that FP16 requires CUDA.`
          Defaults to ``'disk_fp16'`` if ``device`` is CUDA, and ``'disk'`` if CPU.
        device: Device to run inference on.
    """

    MODEL_URLS: ClassVar[dict[str, str]] = {
        "disk": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/disk_lightglue_fused.onnx",
        "superpoint": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/superpoint_lightglue_fused.onnx",
        "disk_fp16": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/disk_lightglue_fused_fp16.onnx",
        "superpoint_fp16": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/superpoint_lightglue_fused_fp16.onnx",
    }

    required_data_keys: ClassVar[list[str]] = ["image0", "image1"]

    def __init__(self, weights: str | None = None, device: Device = "cpu") -> None:
        KORNIA_CHECK(ort is not None, "onnxruntime is not installed.")
        KORNIA_CHECK(np is not None, "numpy is not installed.")

        device = torch.device(device)  # type: ignore
        self.device = device

        if device.type == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise ValueError(f"Unsupported device {device}")

        if weights is None:
            weights = "disk_fp16" if device.type == "cuda" else "disk"

        if weights in self.MODEL_URLS:
            if "fp16" in weights:
                KORNIA_CHECK(device.type == "cuda", "FP16 requires CUDA.")

            url = self.MODEL_URLS[weights]
            if device.type == "cpu":
                url = url.replace(".onnx", "_cpu.onnx")

            weights = download_onnx_from_url(url)

        self.session = ort.InferenceSession(weights, providers=providers)

    def __call__(self, data: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
        return self.forward(data)

    def forward(self, data: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
        r"""Match keypoints and descriptors between two images.

        The output contains the matches (the indices of the matching keypoint pairs between the first and second image)
        and the corresponding confidence scores.
        Only a batch size of 1 is supported.

        Args:
            data: Dictionary containing both images and the keypoints and descriptors thereof.

        Returns:
            Dictionary containing the matches and scores.

        ``data`` (``dict``):
            ``image0`` (``dict``):
                ``keypoints`` (`float32`): :math:`(1, M, 2)`

                ``descriptors`` (`float32`): :math:`(1, M, D)`

                ``image``: :math:`(1, C, H, W)` or ``image_size``: :math:`(1, 2)`

            ``image1`` (``dict``):
                ``keypoints`` (`float32`): :math:`(1, N, 2)`

                ``descriptors`` (`float32`): :math:`(1, N, D)`

                ``image``: :math:`(1, C, H, W)` or ``image_size``: :math:`(1, 2)`

        ``output`` (``dict``):
            ``matches`` (`int64`): :math:`(S, 2)`

            ``scores`` (`float32`): :math:`(S)`
        """
        # Input validation.
        for key in self.required_data_keys:
            KORNIA_CHECK(key in data, f"Missing key {key} in data")
        data0, data1 = data["image0"], data["image1"]
        kpts0_, kpts1_ = data0["keypoints"].contiguous(), data1["keypoints"].contiguous()
        desc0, desc1 = data0["descriptors"].contiguous(), data1["descriptors"].contiguous()
        KORNIA_CHECK_SAME_DEVICES([kpts0_, desc0, kpts1_, desc1], "Wrong device")
        KORNIA_CHECK(kpts0_.device.type == self.device.type, "Wrong device")
        KORNIA_CHECK(torch.float32 == kpts0_.dtype == kpts1_.dtype == desc0.dtype == desc1.dtype, "Wrong dtype")
        KORNIA_CHECK_SHAPE(kpts0_, ["1", "M", "2"])
        KORNIA_CHECK_SHAPE(kpts1_, ["1", "N", "2"])
        KORNIA_CHECK_SHAPE(desc0, ["1", "M", "D"])
        KORNIA_CHECK_SHAPE(desc1, ["1", "N", "D"])
        KORNIA_CHECK(kpts0_.shape[1] == desc0.shape[1], "Number of keypoints does not match number of descriptors")
        KORNIA_CHECK(kpts1_.shape[1] == desc1.shape[1], "Number of keypoints does not match number of descriptors")
        KORNIA_CHECK(desc0.shape[2] == desc1.shape[2], "Descriptors' dimensions do not match")

        # Normalize keypoints.
        size0, size1 = data0.get("image_size"), data1.get("image_size")
        size0 = size0 if size0 is not None else data0["image"].shape[-2:][::-1]  # type: ignore
        size1 = size1 if size1 is not None else data1["image"].shape[-2:][::-1]  # type: ignore

        kpts0 = normalize_keypoints(kpts0_, size=size0)  # type: ignore
        kpts1 = normalize_keypoints(kpts1_, size=size1)  # type: ignore

        KORNIA_CHECK(torch.all(kpts0 >= -1).item() and torch.all(kpts0 <= 1).item(), "")  # type: ignore
        KORNIA_CHECK(torch.all(kpts1 >= -1).item() and torch.all(kpts1 <= 1).item(), "")  # type: ignore

        # Inference.
        lightglue_inputs = {"kpts0": kpts0, "kpts1": kpts1, "desc0": desc0, "desc1": desc1}
        lightglue_outputs = ["matches0", "mscores0"]
        binding = self.session.io_binding()

        for name, tensor in lightglue_inputs.items():
            binding.bind_input(
                name,
                device_type=self.device.type,
                device_id=0,
                element_type=np.float32,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        for name in lightglue_outputs:
            binding.bind_output(name, device_type=self.device.type, device_id=0)

        self.session.run_with_iobinding(binding)

        matches, mscores = binding.get_outputs()

        # TODO: The following is an unnecessary copy. Replace with a better solution when torch supports
        # constructing a tensor from a data pointer, or when ORT supports converting to torch tensor.
        # https://github.com/microsoft/onnxruntime/issues/15963
        outputs = {
            "matches": torch.from_dlpack(matches.numpy()).to(self.device),
            "scores": torch.from_dlpack(mscores.numpy()).to(self.device),
        }
        return outputs
