# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import importlib.util
from typing import ClassVar, Union

import torch

from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_SAME_DEVICES, KORNIA_CHECK_SHAPE
from kornia.core.external import numpy as _np
from kornia.core.external import onnxruntime as _ort

from .utils import download_onnx_from_url, normalize_keypoints

_KEYPOINT_RANGE_MSG = (
    "Keypoints must lie inside the image: normalized by the image size they have to be in [-1, 1]. The keypoints are "
    "(x, y) pixel coordinates. The image size is (width, height) for the ``image_size`` of LightGlue and "
    "OnnxLightGlue, and (height, width) for the ``hw1`` / ``hw2`` of LightGlueMatcher; the wrong order, or keypoints "
    "detected on a differently sized image, put them out of range."
)

__all__ = ["OnnxLightGlue"]


class OnnxLightGlue:
    r"""Match the keypoints and descriptors of one image pair with an exported LightGlue ONNX model.

    ``matcher = OnnxLightGlue("disk", device="cpu")`` downloads an exported LightGlue :cite:`LightGlue2023` model,
    or loads your own ONNX file, into an onnxruntime session, and ``out = matcher(data)`` takes the dicts of
    :class:`LightGlue` for a batch of one: float32 ``keypoints`` ``(1, M, 2)`` in pixels and ``descriptors``
    ``(1, M, D)`` on that device, with ``image_size`` ``(1, 2)`` as ``(W, H)`` or an ``image``. It returns the
    matched index pairs ``matches`` ``(S, 2)`` and their ``scores`` ``(S,)``. It needs ``onnxruntime`` and runs on
    CPU or CUDA only. For batches, other devices or dtypes, scales, orientations or LAFs, or the full outputs of the
    network, use :class:`LightGlue` instead.

    If you have trained your own LightGlue model, see https://github.com/fabio-sim/LightGlue-ONNX
    for how to export the model to ONNX and optimize it.

    Convention:
        - See the Convention block on :class:`~kornia.feature.LightGlue`: the keypoints are normalised the same way,
          with ``image_size`` in the order ``(W, H)``, and the same range check applies. Unlike
          :class:`~kornia.feature.LightGlue`, it takes float32 inputs with a batch of one and no scales, orientations
          or LAFs, and it runs on CPU or CUDA only.
        - A CUDA ``device`` names one GPU, and the session runs on it: ``"cuda:1"`` runs on GPU 1, and ``"cuda"``
          without an index means GPU 0, which the ``device`` attribute then reports as ``cuda:0``. The inputs must
          be on that same GPU; tensors on another GPU raise.

    Args:
        weights: Pretrained weights, or a path to your own exported ONNX model. Available pretrained weights
          are ``'disk'``, ``'superpoint'``, ``'disk_fp16'``, and ``'superpoint_fp16'``. `Note that FP16 requires CUDA.`
          Defaults to ``'disk_fp16'`` if ``device`` is CUDA, and ``'disk'`` if CPU.
        device: ``'cpu'`` or ``'cuda'`` to run inference on, with an optional GPU index such as ``'cuda:1'``;
          another device type raises ``ValueError``. ``None`` means ``'cpu'``, as in the other kornia feature
          constructors, not torch's current default device.

    See Also:
        - :class:`LightGlue`: the PyTorch network, which also runs batches, on MPS, and in float16, bfloat16 or
          float64 (see its Convention block), and returns every output.
        - :class:`LightGlueMatcher`: LightGlue on descriptors and LAFs, returning scores and index pairs like the
          other kornia matchers.

    """

    MODEL_URLS: ClassVar[dict[str, str]] = {
        "disk": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/disk_lightglue_fused.onnx",
        "superpoint": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/superpoint_lightglue_fused.onnx",
        "disk_fp16": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/disk_lightglue_fused_fp16.onnx",
        "superpoint_fp16": "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/superpoint_lightglue_fused_fp16.onnx",
    }

    required_data_keys: ClassVar[list[str]] = ["image0", "image1"]

    def __init__(self, weights: str | None = None, device: Union[str, torch.device, None] = "cpu") -> None:
        # Import onnxruntime through the lazy loader, not an `importlib.util.find_spec` probe, so that
        # `kornia_config.lazyloader.installation_mode` handles a missing module before the weights are downloaded.
        # `__version__` is module metadata, which the loader imports the module for; `__doc__` would not import it.
        _ = _ort.__version__
        KORNIA_CHECK(importlib.util.find_spec("numpy") is not None, "numpy is not installed.")

        device = torch.device("cpu") if device is None else torch.device(device)

        providers: list[str | tuple[str, dict[str, int]]]
        if device.type == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device.type == "cuda":
            # onnxruntime's CUDA provider and the I/O bindings both take a GPU index; without one they use GPU 0.
            device = torch.device("cuda", 0 if device.index is None else device.index)
            providers = [("CUDAExecutionProvider", {"device_id": device.index}), "CPUExecutionProvider"]
        else:
            raise ValueError(f"Unsupported device {device}")
        self.device = device

        if weights is None:
            weights = "disk_fp16" if device.type == "cuda" else "disk"

        if weights in self.MODEL_URLS:
            if "fp16" in weights:
                KORNIA_CHECK(device.type == "cuda", "FP16 requires CUDA.")

            url = self.MODEL_URLS[weights]
            if device.type == "cpu":
                url = url.replace(".onnx", "_cpu.onnx")

            weights = download_onnx_from_url(url)

        self.session = _ort.InferenceSession(weights, providers=providers)

    def __call__(self, data: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Run the ONNX LightGlue matcher.

        Args:
            data: Dictionary containing image features, keypoints, descriptors, and image sizes for matching.

        Returns:
            Dictionary containing ONNX LightGlue matches and matching scores.
        """
        return self.forward(data)

    def forward(self, data: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
                ``keypoints`` (`float32`): ``(1, M, 2)``

                ``descriptors`` (`float32`): ``(1, M, D)``

                ``image``: ``(1, C, H, W)`` or ``image_size``: ``(1, 2)``

            ``image1`` (``dict``):
                ``keypoints`` (`float32`): ``(1, N, 2)``

                ``descriptors`` (`float32`): ``(1, N, D)``

                ``image``: ``(1, C, H, W)`` or ``image_size``: ``(1, 2)``

        ``output`` (``dict``):
            ``matches`` (`int64`): ``(S, 2)``

            ``scores`` (`float32`): ``(S,)``

        """
        # Input validation.
        for key in self.required_data_keys:
            KORNIA_CHECK(key in data, f"Missing key {key} in data")
        data0, data1 = data["image0"], data["image1"]
        kpts0_, kpts1_ = data0["keypoints"].contiguous(), data1["keypoints"].contiguous()
        desc0, desc1 = data0["descriptors"].contiguous(), data1["descriptors"].contiguous()
        KORNIA_CHECK_SAME_DEVICES([kpts0_, desc0, kpts1_, desc1], "Wrong device")
        KORNIA_CHECK(
            kpts0_.device == self.device,
            f"Wrong device: the inputs are on {kpts0_.device}, the matcher on {self.device}",
        )
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
        size0 = size0 if size0 is not None else data0["image"].shape[-2:][::-1]
        size1 = size1 if size1 is not None else data1["image"].shape[-2:][::-1]

        kpts0 = normalize_keypoints(kpts0_, size=size0)
        kpts1 = normalize_keypoints(kpts1_, size=size1)

        KORNIA_CHECK(torch.all(kpts0 >= -1).item() and torch.all(kpts0 <= 1).item(), _KEYPOINT_RANGE_MSG)
        KORNIA_CHECK(torch.all(kpts1 >= -1).item() and torch.all(kpts1 <= 1).item(), _KEYPOINT_RANGE_MSG)

        # Inference.
        lightglue_inputs = {"kpts0": kpts0, "kpts1": kpts1, "desc0": desc0, "desc1": desc1}
        lightglue_outputs = ["matches0", "mscores0"]
        binding = self.session.io_binding()
        device_id = 0 if self.device.index is None else self.device.index

        for name, tensor in lightglue_inputs.items():
            binding.bind_input(
                name,
                device_type=self.device.type,
                device_id=device_id,
                element_type=_np.float32,
                shape=tuple(tensor.shape),
                buffer_ptr=tensor.data_ptr(),
            )

        for name in lightglue_outputs:
            binding.bind_output(name, device_type=self.device.type, device_id=device_id)

        self.session.run_with_iobinding(binding)

        matches, mscores = binding.get_outputs()
        # Prefer DLPack-based conversion when available for zero-copy transfer between them
        # The fallback path uses NumPy, which incurs a device-to-host copy and is slower.

        # ORT's io_binding.get_outputs() returns Python-level OrtValue wrappers.
        # The DLPack protocol (__dlpack__ + __dlpack_device__) is only available on
        # the underlying pybind11 C object (_ortvalue), not the Python wrapper itself.
        # We use getattr to handle both current ORT (needs ._ortvalue) and future ORT
        # versions that may expose __dlpack__ directly on the wrapper.
        # Both outputs must support the full protocol before taking the zero-copy path.
        _m = getattr(matches, "_ortvalue", matches)
        _s = getattr(mscores, "_ortvalue", mscores)
        if (
            hasattr(_m, "__dlpack__")
            and hasattr(_m, "__dlpack_device__")
            and hasattr(_s, "__dlpack__")
            and hasattr(_s, "__dlpack_device__")
        ):
            outputs = {
                "matches": torch.from_dlpack(_m),
                "scores": torch.from_dlpack(_s),
            }
        else:
            outputs = {
                "matches": torch.from_numpy(matches.numpy()).to(self.device),
                "scores": torch.from_numpy(mscores.numpy()).to(self.device),
            }
        return outputs
