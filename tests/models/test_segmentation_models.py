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

import pytest
import torch
from torch import nn

from kornia.models.segmentation import SegmentationModelsBuilder, SemanticSegmentation

from testing.base import BaseTester

# The dict shape `smp.encoders.get_preprocessing_params("resnet34")` returns.
IMAGENET_PARAMS = {
    "input_space": "RGB",
    "input_range": [0, 1],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def _stand_in_network(classes: int = 2) -> nn.Module:
    """A tiny (B, 3, H, W) -> (B, classes, H, W) network standing in for an smp model."""
    return nn.Sequential(nn.Conv2d(3, classes, kernel_size=1), nn.Softmax(dim=1))


class TestSegmentationModelsBuilder(BaseTester):
    """`build` wraps a user-built network; kornia imports no segmentation package for it."""

    def test_smoke(self, device, dtype):
        net = _stand_in_network()
        model = SegmentationModelsBuilder.build(net, IMAGENET_PARAMS, name="stand_in")
        assert isinstance(model, SemanticSegmentation)
        assert model.name == "stand_in"
        assert model.model is net
        assert not model.model.training
        assert isinstance(model.post_processor, nn.Identity)

        model = model.to(device, dtype)
        images = torch.rand(2, 3, 8, 12, device=device, dtype=dtype)
        out = model(images)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 2, 8, 12)
        assert out.device == images.device and out.dtype == dtype
        assert torch.isfinite(out).all()
        # The softmax head normalizes over the class axis, so the output is not a degenerate map.
        self.assert_close(out.sum(dim=1), torch.ones(2, 8, 12, device=device, dtype=dtype))

    def test_list_input(self, device, dtype):
        model = SegmentationModelsBuilder.build(_stand_in_network(), IMAGENET_PARAMS).to(device, dtype)
        images = [torch.rand(3, 8, 12, device=device, dtype=dtype), torch.rand(3, 6, 6, device=device, dtype=dtype)]
        out = model(images)
        assert isinstance(out, list) and len(out) == 2
        assert out[0].shape == (2, 8, 12) and out[1].shape == (2, 6, 6)
        self.assert_close(out[0], model(images[0][None])[0])

    def test_default_name(self):
        assert SegmentationModelsBuilder.build(_stand_in_network()).name == "segmentation_model"

    def test_none_params_is_identity_preprocessing(self, device, dtype):
        model = SegmentationModelsBuilder.build(_stand_in_network(), preproc_params=None).to(device, dtype)
        assert isinstance(model.pre_processor, nn.Identity)
        images = torch.rand(1, 3, 5, 7, device=device, dtype=dtype)
        self.assert_close(model(images), model.model(images))

    def test_preprocessing_bgr_255(self, device, dtype):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        params = {"input_space": "BGR", "input_range": [0, 255], "mean": mean, "std": std}
        pipeline = SegmentationModelsBuilder.get_preprocessing_pipeline(params).to(device, dtype)

        x = torch.rand(2, 3, 5, 7, device=device, dtype=dtype)
        out = pipeline(x)

        # By hand, in float64: flip the channels, rescale [0, 1] -> [0, 255], then (x - mean) / std per channel.
        # The reference lives on the CPU because MPS has no float64; the comparison brings `out` over to it.
        mean_t = torch.tensor(mean, dtype=torch.float64).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float64).view(1, 3, 1, 1)
        x64 = x.cpu().to(torch.float64)
        expected = (x64.flip(1) * 255.0 - mean_t) / std_t
        out64 = out.cpu().to(torch.float64)

        assert out.shape == x.shape and out.dtype == dtype and out.device == x.device
        # The pipeline chains three rounding ops in the working dtype at magnitudes up to ~1100, so the
        # accumulated error reaches ~2 eps (measured over 50 draws: 1.9 float32, 1.3 float16, 1.4 bfloat16)
        # while BaseTester's half-precision defaults are rtol ~1 eps and fail here. The pipeline builds its
        # mean/std constants as float32 tensors (float64 ones would change the ONNX graph's output dtype),
        # so a float64 input is only float32-accurate (measured 1.3 float32 eps); floor the tolerance there.
        # Pin the exact math within 8 eps of the coarser of the two.
        tol = 8 * max(torch.finfo(dtype).eps, torch.finfo(torch.float32).eps)
        self.assert_close(out64, expected, rtol=tol, atol=tol)
        # The flip is load-bearing: the un-flipped normalization is a different tensor.
        assert not torch.allclose(out64, (x64 * 255.0 - mean_t) / std_t, rtol=tol, atol=tol)

    def test_preprocessing_255_rescale_is_exact(self, device, dtype):
        # The [0, 1] -> [0, 255] step multiplies by 255, which every dtype stores exactly. A division by
        # the reciprocal is not: bfloat16 stores 1/255 as 0.0039368 and maps 0.5 to 127.0 (127.5 is
        # representable in every dtype here, so equality is the right check).
        params = {"input_space": "RGB", "input_range": [0, 255], "mean": None, "std": None}
        pipeline = SegmentationModelsBuilder.get_preprocessing_pipeline(params).to(device, dtype)
        x = torch.tensor([0.0, 0.5, 1.0], device=device, dtype=dtype).view(1, 3, 1, 1).expand(1, 3, 2, 2)
        expected = torch.tensor([0.0, 127.5, 255.0], device=device, dtype=dtype).view(1, 3, 1, 1).expand(1, 3, 2, 2)
        out = pipeline(x)
        assert out.dtype == dtype and out.device == x.device
        assert torch.equal(out, expected)

    def test_preprocessing_rgb_unit_range(self, device, dtype):
        pipeline = SegmentationModelsBuilder.get_preprocessing_pipeline(IMAGENET_PARAMS).to(device, dtype)
        x = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        mean_t = torch.tensor(IMAGENET_PARAMS["mean"], device=device, dtype=dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(IMAGENET_PARAMS["std"], device=device, dtype=dtype).view(1, 3, 1, 1)
        self.assert_close(pipeline(x), (x - mean_t) / std_t)

    def test_preprocessing_no_mean_std_is_identity(self, device, dtype):
        params = {"input_space": "RGB", "input_range": [0, 1], "mean": None, "std": None}
        pipeline = SegmentationModelsBuilder.get_preprocessing_pipeline(params).to(device, dtype)
        x = torch.rand(1, 3, 4, 4, device=device, dtype=dtype)
        self.assert_close(pipeline(x), x)

    @pytest.mark.parametrize("missing", ["input_space", "input_range", "mean", "std"])
    def test_exception_missing_key(self, missing):
        params = {k: v for k, v in IMAGENET_PARAMS.items() if k != missing}
        # Match the KORNIA_CHECK message, not just the key: a bare `KeyError(key)` would match the key too.
        with pytest.raises(Exception, match=f"preproc_params is missing the key '{missing}'"):
            SegmentationModelsBuilder.get_preprocessing_pipeline(params)

    def test_exception_unsupported_input_space(self):
        params = {**IMAGENET_PARAMS, "input_space": "HSV"}
        with pytest.raises(ValueError, match="Unsupported input space: HSV"):
            SegmentationModelsBuilder.get_preprocessing_pipeline(params)

    def test_exception_unsupported_input_range(self):
        params = {**IMAGENET_PARAMS, "input_range": [0, 65535]}
        with pytest.raises(ValueError, match="Unsupported input range"):
            SegmentationModelsBuilder.get_preprocessing_pipeline(params)

    def test_exception_missing_key_with_build(self):
        params = {k: v for k, v in IMAGENET_PARAMS.items() if k != "std"}
        with pytest.raises(Exception, match="preproc_params is missing the key 'std'"):
            SegmentationModelsBuilder.build(_stand_in_network(), params)


class TestSemanticSegmentation(BaseTester):
    def test_from_config_not_implemented(self):
        with pytest.raises(NotImplementedError, match=r"SegmentationModelsBuilder\.build"):
            SemanticSegmentation.from_config(None)

    def test_visualize(self, device, dtype):
        model = SegmentationModelsBuilder.build(_stand_in_network(3), IMAGENET_PARAMS).to(device, dtype)
        images = torch.rand(2, 3, 6, 6, device=device, dtype=dtype)
        vis = model.visualize(images)
        assert vis.shape == (2, 3, 6, 6)
        # The colormap is drawn on the CPU in float32; the gather has to happen on the mask's device (a CUDA
        # or MPS mask indexing a CPU colormap raised before) and the result keeps the model's device and dtype.
        # The softmax probe tolerates a half-precision sum of probabilities, which allclose's default did not.
        assert vis.device == images.device
        assert vis.dtype == dtype
        assert torch.isfinite(vis).all()
        # Same through the per-image list path, which draws a colormap per mask.
        vis_list = model.visualize([images[0], images[1]])
        assert isinstance(vis_list, list) and len(vis_list) == 2
        assert vis_list[0].shape == (3, 6, 6) and vis_list[0].device == images.device
        self.assert_close(vis_list[0], vis[0])

    def test_visualize_rejects_logits(self, device, dtype):
        # The probe's tolerance scales with the dtype, but a raw-logit head (no softmax) must still be refused.
        logits_net = nn.Conv2d(3, 3, kernel_size=1)
        with torch.no_grad():
            logits_net.weight.mul_(4.0)
        model = SegmentationModelsBuilder.build(logits_net, IMAGENET_PARAMS).to(device, dtype)
        with pytest.raises(ValueError, match="softmax"):
            model.visualize(torch.rand(1, 3, 6, 6, device=device, dtype=dtype))

    def test_preprocessing_onnx_export(self, device):
        # The pipeline is documented as ONNX-friendly: export it and compare the runtime against eager.
        pytest.importorskip("onnx")
        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("onnxscript")
        if device.type != "cpu":
            # `Normalize` keeps mean/std as plain tensor attributes that `.to(device)` does not move, so
            # exporting an accelerator-resident pipeline fails on a device mismatch (pre-existing).
            pytest.skip("export of a non-CPU pipeline is blocked by Normalize's tensor attributes")
        params = {**IMAGENET_PARAMS, "input_space": "BGR", "input_range": [0, 255]}
        pipeline = SegmentationModelsBuilder.get_preprocessing_pipeline(params).to(device).eval()
        x = torch.rand(1, 3, 6, 6, device=device)
        program = torch.onnx.export(pipeline, (x,), dynamo=True, opset_version=18, verbose=False)
        assert program is not None
        session = ort.InferenceSession(program.model_proto.SerializeToString(), providers=["CPUExecutionProvider"])
        (out,) = session.run(None, {session.get_inputs()[0].name: x.cpu().numpy()})
        self.assert_close(torch.from_numpy(out), pipeline(x).cpu(), rtol=1e-5, atol=1e-5)
