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

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

from kornia.models.depth_estimation import DepthEstimation  # noqa: E402

from testing.base import BaseTester  # noqa: E402

# onnx-community/depth-anything-v2-small preprocessor_config.json (fetched 2026-10-16)
_DEPTH_ANYTHING_V2_SMALL_CONFIG = {
    "do_normalize": True,
    "do_pad": False,
    "do_rescale": True,
    "do_resize": True,
    "ensure_multiple_of": 14,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "DPTImageProcessor",
    "image_std": [0.229, 0.224, 0.225],
    "keep_aspect_ratio": True,
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 518, "width": 518},
    "size_divisor": None,
}


def _stand_in_graph() -> onnx.ModelProto:
    """Return a graph with Depth-Anything's I/O contract, ``(B, 3, H, W) -> (B, H, W)``: the mean over channels.

    The real onnx-community graph is not downloaded. ``axes`` is an input of ``ReduceMean`` from opset 18.
    """
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["input", "axes"], ["output"], keepdims=0)],
        "stand_in_depth",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["B", 3, "H", "W"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["B", "H", "W"])],
        initializer=[axes],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _depth_anything_resize():
    from kornia.models._hf_models.preprocessor import _DPTResize

    return _DPTResize(size=(518, 518), keep_aspect_ratio=True, multiple=14, interpolation="bicubic")


class _SessionSpy:
    """Wrap an ONNX Runtime session and keep every array the graph receives and returns."""

    def __init__(self, session) -> None:
        self.session = session
        self.inputs: list = []
        self.outputs: list = []

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, output_names, feeds):
        self.inputs.extend(feeds.values())
        outputs = self.session.run(output_names, feeds)
        self.outputs.extend(outputs)
        return outputs


def _spy_on(model) -> _SessionSpy:
    spy = _SessionSpy(model.get_session())
    model.set_session(spy)
    return spy


@pytest.mark.device_agnostic
@pytest.mark.parametrize(
    "height, width, size, keep_aspect_ratio, expected",
    [
        # transformers DPTImageProcessor (size 518x518, ensure_multiple_of 14), the issue #6001 table. The side whose
        # scale is closer to 1 is fitted; the other keeps the aspect ratio and is rounded to a multiple of 14:
        # 518/400 = 1.295 beats 518/300 = 1.727, so H = round(300 * 1.295 / 14) * 14 = round(27.75) * 14 = 392
        (300, 400, (518, 518), True, (392, 518)),
        # 518/480 = 1.079 beats 518/640 = 0.809, so W = round(640 * 1.079 / 14) * 14 = round(49.33) * 14 = 686
        (480, 640, (518, 518), True, (518, 686)),
        # 518/1080 = 0.480 beats 518/1920 = 0.270, so W = round(1920 * 0.480 / 14) * 14 = round(65.78) * 14 = 924
        (1080, 1920, (518, 518), True, (518, 924)),
        # 518/500 = 1.036 beats 518/1000 = 0.518, so H = round(1000 * 1.036 / 14) * 14 = 74 * 14 = 1036
        (1000, 500, (518, 518), True, (1036, 518)),
        # without keep_aspect_ratio each side is scaled to its target: 518 = 37 * 14
        (300, 400, (518, 518), False, (518, 518)),
        # Python round sends a tie to the even integer: 35 / 14 = 2.5 -> 2 -> 28, and 21 / 14 = 1.5 -> 2 -> 28
        (10, 10, (35, 21), False, (28, 28)),
        # 518/296 = 1.75 and 518/2072 = 0.25 are both 0.75 from 1: the strict < fits the height, so
        # W = 2072 * 1.75 = 3626 = 259 * 14
        (296, 2072, (518, 518), True, (518, 3626)),
        # 518/1000 = 0.518 beats 518/10 = 51.8, so H = round(10 * 0.518 / 14) * 14 = round(0.37) * 14 = 0, which is
        # kept at one multiple, 14 (transformers would pass 0 to the resize and raise); and the transposed image
        (10, 1000, (518, 518), True, (14, 518)),
        (1000, 10, (518, 518), True, (518, 14)),
    ],
)
def test_dpt_output_size(height, width, size, keep_aspect_ratio, expected):
    from kornia.models._hf_models.preprocessor import _dpt_output_size

    assert _dpt_output_size(height, width, size, keep_aspect_ratio, 14) == expected


@pytest.mark.device_agnostic
def test_from_json_keeps_the_aspect_ratio():
    from kornia.models._hf_models.preprocessor import DPTImageProcessor

    pipeline = DPTImageProcessor.from_json(dict(_DEPTH_ANYTHING_V2_SMALL_CONFIG))

    assert pipeline(torch.rand(1, 3, 300, 400)).shape[-2:] == (392, 518)
    # resample 3 is PIL's BICUBIC
    assert pipeline[0].interpolation == "bicubic"


@pytest.mark.device_agnostic
@pytest.mark.parametrize(
    "dropped, changes, expected",
    [
        # absent keep_aspect_ratio and ensure_multiple_of take DPTImageProcessor's defaults, False and 1: each side is
        # scaled to 500, which a multiple of 14 would round to 504
        (("keep_aspect_ratio", "ensure_multiple_of"), {"size": {"height": 500, "width": 500}}, (500, 500)),
        # a non-square size is (height, width): 518 = 37 * 14 and 700 = 50 * 14
        ((), {"keep_aspect_ratio": False, "size": {"height": 518, "width": 700}}, (518, 700)),
    ],
)
def test_from_json_reads_the_size_and_the_defaults(dropped, changes, expected):
    from kornia.models._hf_models.preprocessor import DPTImageProcessor

    config = {k: v for k, v in _DEPTH_ANYTHING_V2_SMALL_CONFIG.items() if k not in dropped}
    config.update(changes)

    assert DPTImageProcessor.from_json(config)(torch.rand(1, 3, 300, 400)).shape[-2:] == expected


@pytest.mark.device_agnostic
@pytest.mark.parametrize(
    "resample, interpolation", [(0, "nearest-exact"), (2, "bilinear"), (3, "bicubic"), (None, "bicubic")]
)
def test_from_json_maps_resample_to_the_interpolation(resample, interpolation):
    # PIL's NEAREST (0), which samples the pixel centres, is torch's "nearest-exact"; BILINEAR (2) and BICUBIC (3).
    # An absent key is DPTImageProcessor's default, BICUBIC
    from kornia.models._hf_models.preprocessor import DPTImageProcessor

    config = {k: v for k, v in _DEPTH_ANYTHING_V2_SMALL_CONFIG.items() if k != "resample"}
    if resample is not None:
        config["resample"] = resample

    assert DPTImageProcessor.from_json(config)[0].interpolation == interpolation


@pytest.mark.device_agnostic
def test_dpt_resize_nearest_matches_pil():
    # PIL 12.3.0 NEAREST samples the pixel centres, without a filter:
    #   np.asarray(Image.fromarray(np.arange(24).reshape(4, 6).astype(np.uint8)).resize((3, 2), Image.NEAREST))
    from kornia.models._hf_models.preprocessor import _DPTResize

    image = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
    out = _DPTResize(size=(2, 3), keep_aspect_ratio=False, multiple=1, interpolation="nearest-exact")(image)

    BaseTester.assert_close(out, torch.tensor([[[[7.0, 9.0, 11.0], [19.0, 21.0, 23.0]]]]), rtol=0.0, atol=0.0)


# PIL 12.3.0, the filter of transformers' slow DPTImageProcessor, on mid-range values, so its uint8 passes never clip an
# overshoot: src = 64 + np.arange(h * w).reshape(h, w) * 53 % 128, then
#   np.asarray(Image.fromarray(src.astype(np.uint8)).resize((out_w, out_h), Image.BICUBIC or Image.BILINEAR))
_PIL_DOWNSCALE_8X12_TO_4X6 = {
    "bicubic": [
        [129, 133, 101, 137, 129, 112],
        [151, 121, 124, 137, 120, 101],
        [141, 110, 146, 136, 119, 126],
        [133, 102, 136, 127, 119, 150],
    ],
    "bilinear": [
        [131, 130, 108, 134, 128, 112],
        [147, 123, 125, 135, 121, 105],
        [139, 115, 141, 135, 121, 125],
        [132, 108, 134, 128, 122, 145],
    ],
}
_PIL_BICUBIC_UPSCALE_3X4_TO_6X8 = [
    [51, 68, 103, 138, 171, 158, 105, 80],
    [78, 85, 98, 121, 156, 155, 122, 107],
    [138, 121, 85, 85, 123, 148, 160, 166],
    [149, 133, 99, 88, 103, 128, 164, 181],
    [112, 121, 138, 130, 97, 99, 133, 150],
    [94, 114, 156, 150, 94, 84, 119, 136],
]


def _pil_source(height: int, width: int) -> torch.Tensor:
    return (64 + torch.arange(height * width).reshape(1, 1, height, width) * 53 % 128).float() / 255


@pytest.mark.device_agnostic
@pytest.mark.parametrize("interpolation", ["bicubic", "bilinear"])
def test_dpt_resize_downscale_matches_pil(interpolation):
    # 4/8 = 0.5 beats 4/12 = 0.33, so the 8x12 image is halved to 4x6. PIL widens its kernel on a downscale, which the
    # antialiased interpolate matches to within its uint8 rounding (max 0.62/255 measured); without antialiasing the
    # bicubic is up to 27/255 off. The tolerance is 1/255.
    from kornia.models._hf_models.preprocessor import _DPTResize

    resize = _DPTResize(size=(4, 4), keep_aspect_ratio=True, multiple=2, interpolation=interpolation)

    out = resize(_pil_source(8, 12))

    expected = torch.tensor(_PIL_DOWNSCALE_8X12_TO_4X6[interpolation], dtype=torch.float32)[None, None] / 255
    BaseTester.assert_close(out, expected, rtol=0.0, atol=1 / 255)


@pytest.mark.device_agnostic
def test_dpt_resize_bicubic_upscale_matches_pil():
    # PIL's bicubic has a = -0.5, as the antialiased interpolate does (max 0.78/255 measured); the plain bicubic has
    # a = -0.75 and is up to 6.2/255 off. The tolerance is 1/255.
    from kornia.models._hf_models.preprocessor import _DPTResize

    resize = _DPTResize(size=(6, 8), keep_aspect_ratio=False, multiple=1, interpolation="bicubic")

    out = resize(_pil_source(3, 4))

    expected = torch.tensor(_PIL_BICUBIC_UPSCALE_3X4_TO_6X8, dtype=torch.float32)[None, None] / 255
    BaseTester.assert_close(out, expected, rtol=0.0, atol=1 / 255)


@pytest.mark.device_agnostic
@pytest.mark.parametrize("resample", [1, 4, 5])
def test_from_json_rejects_an_unsupported_resample(resample):
    # PIL's LANCZOS (1), BOX (4) and HAMMING (5) have no kornia resize mode
    from kornia.models._hf_models.preprocessor import DPTImageProcessor

    with pytest.raises(NotImplementedError, match="resample"):
        DPTImageProcessor.from_json({**_DEPTH_ANYTHING_V2_SMALL_CONFIG, "resample": resample})


class TestDepthEstimation(BaseTester):
    @staticmethod
    def _skip_float64_on_mps(device, dtype) -> None:
        if device.type == "mps" and dtype == torch.float64:
            pytest.skip("MPS does not support float64")

    def test_tensor_batch_returns_b1hw(self, device, dtype):
        self._skip_float64_on_mps(device, dtype)
        model = DepthEstimation(_stand_in_graph(), pre_processor=None)
        images = torch.rand(2, 3, 300, 400, device=device, dtype=dtype)

        depth = model(images)

        assert depth.shape == (2, 1, 300, 400)
        # the stand-in's depth is the channel mean; without a resize stage it is computed at the input size
        self.assert_close(depth, images.float().mean(1, keepdim=True).to(dtype))

    @pytest.mark.parametrize("sizes", [[(300, 400), (200, 150)], [(200, 150)], []])
    def test_list_returns_1hw_items(self, device, dtype, sizes):
        self._skip_float64_on_mps(device, dtype)
        model = DepthEstimation(_stand_in_graph(), pre_processor=None)
        images = [torch.rand(3, h, w, device=device, dtype=dtype) for h, w in sizes]

        depths = model(images)

        assert isinstance(depths, list)
        assert [tuple(d.shape) for d in depths] == [(1, h, w) for h, w in sizes]
        for depth, image in zip(depths, images):
            self.assert_close(depth, image.float().mean(0, keepdim=True).to(dtype))

    @pytest.mark.parametrize("as_list", [False, True])
    def test_visualize_and_save_take_a_tensor_batch(self, device, dtype, as_list, tmp_path):
        self._skip_float64_on_mps(device, dtype)
        model = DepthEstimation(_stand_in_graph(), pre_processor=None)
        images = torch.rand(2, 3, 300, 400, device=device, dtype=dtype)
        if as_list:
            images = list(images)

        views = model.visualize(images)

        assert [tuple(v.shape) for v in views] == [(3, 300, 400), (3, 300, 400)]
        # relative depth: each map divided by its maximum, repeated over three channels
        depth = images[0].float().mean(0, keepdim=True).to(dtype)
        self.assert_close(views[0], (depth / depth.max()).expand(3, -1, -1))

        model.save(images, directory=str(tmp_path))

        names = sorted(p.name for p in tmp_path.iterdir())
        assert len(names) == 4
        assert sum("_src_" in n for n in names) == 2
        assert sum("_depth_" in n for n in names) == 2

    def test_any_float_dtype(self, device, dtype):
        self._skip_float64_on_mps(device, dtype)
        model = DepthEstimation(_stand_in_graph(), pre_processor=None)
        images = torch.rand(2, 3, 30, 40, device=device, dtype=dtype)

        depth = model(images)
        (item,) = model([images[0]])

        for out in (depth, item):
            assert out.dtype == dtype
            assert out.device == images.device
            assert torch.isfinite(out).all()
        # ONNX Runtime computes in float32, and the depth is cast to the input dtype at the end
        self.assert_close(depth, images.float().mean(1, keepdim=True).to(dtype))
        self.assert_close(item, depth[0])

    @pytest.mark.parametrize("as_list", [False, True])
    @pytest.mark.parametrize("int_dtype", [torch.uint8, torch.int64])
    def test_integer_images_raise(self, device, as_list, int_dtype):
        # the float32 cast must not hand the graph a [0, 255] image as if it were a [0, 1] one
        from kornia.core.exceptions import BaseError

        model = DepthEstimation(_stand_in_graph(), pre_processor=None)
        images = torch.randint(0, 256, (1, 3, 30, 40), device=device, dtype=int_dtype)

        with pytest.raises(BaseError, match="floating dtype"):
            model(list(images) if as_list else images)

    def test_graph_sees_the_dpt_size(self, device):
        # the resize runs on the CPU whatever the image's device, with the filter of transformers' DPT image processor
        model = DepthEstimation(_stand_in_graph(), pre_processor=None, resize=_depth_anything_resize())
        spy = _spy_on(model)
        images = torch.rand(2, 3, 300, 400, device=device)

        depth = model(images)

        assert [tuple(x.shape) for x in spy.inputs] == [(2, 3, 392, 518)]
        assert depth.shape == (2, 1, 300, 400)
        assert depth.device == images.device
        expected = torch.nn.functional.interpolate(
            images.cpu(), size=(392, 518), mode="bicubic", align_corners=False, antialias=True
        )
        self.assert_close(torch.from_numpy(spy.inputs[0]), expected, rtol=0.0, atol=0.0)

    def test_depth_is_resized_back_bicubically(self, device):
        # as transformers' DPT post_process_depth_estimation does: interpolate(depth[None, None], size=target_size,
        # mode="bicubic", align_corners=False), here on the CPU in float32
        model = DepthEstimation(_stand_in_graph(), pre_processor=None, resize=_depth_anything_resize())
        spy = _spy_on(model)
        images = torch.rand(2, 3, 300, 400, device=device)

        depth = model(images)

        assert [x.shape for x in spy.outputs] == [(2, 392, 518)]
        expected = torch.nn.functional.interpolate(
            torch.from_numpy(spy.outputs[0])[:, None], size=(300, 400), mode="bicubic", align_corners=False
        )
        self.assert_close(depth.cpu(), expected, rtol=0.0, atol=0.0)

    def test_graph_sees_the_dpt_size_of_each_list_item(self, device):
        model = DepthEstimation(_stand_in_graph(), pre_processor=None, resize=_depth_anything_resize())
        spy = _spy_on(model)
        # 200x150: 518/200 = 2.59 beats 518/150 = 3.45, so W = round(150 * 2.59 / 14) * 14 = round(27.75) * 14 = 392
        images = [torch.rand(3, 300, 400, device=device), torch.rand(3, 200, 150, device=device)]

        depths = model(images)

        assert [tuple(x.shape) for x in spy.inputs] == [(1, 3, 392, 518), (1, 3, 518, 392)]
        assert [tuple(d.shape) for d in depths] == [(1, 300, 400), (1, 200, 150)]

    def test_resize_runs_in_float32(self, device, dtype):
        # the resize runs in float32 whatever the input dtype, so the depth of a half image is the depth of the same
        # values in float32, cast at the end
        self._skip_float64_on_mps(device, dtype)
        model = DepthEstimation(_stand_in_graph(), pre_processor=None, resize=_depth_anything_resize())
        images = torch.rand(1, 3, 60, 80, device=device, dtype=dtype)

        depth = model(images)

        assert depth.dtype == dtype
        self.assert_close(depth, model(images.float()).to(dtype), rtol=0.0, atol=0.0)


@pytest.mark.device_agnostic
def test_to_onnx_graph_takes_a_resized_image():
    from kornia.core import ImageSequential
    from kornia.enhance import Normalize, Rescale

    mean, std = torch.tensor([[0.485, 0.456, 0.406]]), torch.tensor([[0.229, 0.224, 0.225]])
    pre = ImageSequential(Rescale(2.0), Normalize(mean=mean, std=std)).to_onnx(save=False)
    model = DepthEstimation(_stand_in_graph(), pre_processor=pre, resize=_depth_anything_resize())
    spy = _spy_on(model)
    image = torch.rand(1, 3, 300, 400)
    model(image)  # the depth the model computes before resizing it back to 300x400

    combined = model.to_onnx(include_pre_and_post_processor=True, save=False)

    import onnxruntime as ort

    session = ort.InferenceSession(combined.SerializeToString(), providers=["CPUExecutionProvider"])
    resized = model.resize(image)
    assert resized.shape == (1, 3, 392, 518)
    (depth,) = session.run(None, {session.get_inputs()[0].name: resized.numpy()})
    assert depth.shape == (1, 392, 518)
    BaseTester.assert_close(torch.from_numpy(depth), torch.from_numpy(spy.outputs[0]))
    expected = ((resized * 2.0 - mean[..., None, None]) / std[..., None, None]).mean(1)
    BaseTester.assert_close(torch.from_numpy(depth), expected)


@pytest.mark.device_agnostic
def test_builder_resizes_in_torch_before_the_graph(monkeypatch):
    # DepthAnythingONNXBuilder.build with the loader's downloads replaced by the stand-in graph and the
    # depth-anything-v2-small config: the DPT resize runs in torch and the exported preprocessing graph keeps the rest
    from kornia.models._hf_models import HFONNXComunnityModelLoader
    from kornia.models._hf_models.preprocessor import PreprocessingLoader, _DPTResize
    from kornia.models.depth_estimation import DepthAnythingONNXBuilder

    monkeypatch.setattr(HFONNXComunnityModelLoader, "load_model", lambda self, **kwargs: _stand_in_graph())
    monkeypatch.setattr(
        HFONNXComunnityModelLoader,
        "load_preprocessing",
        lambda self: PreprocessingLoader.from_json(dict(_DEPTH_ANYTHING_V2_SMALL_CONFIG)),
    )

    model = DepthAnythingONNXBuilder.build()

    assert isinstance(model.resize, _DPTResize)
    assert model.resize.interpolation == "bicubic"
    spy = _spy_on(model)
    depth = model(torch.rand(1, 3, 300, 400))
    assert [tuple(x.shape) for x in spy.inputs] == [(1, 3, 392, 518)]
    assert depth.shape == (1, 1, 300, 400)
