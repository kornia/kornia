import pytest
import torch

from kornia.feature.dedode import DeDoDe
from kornia.utils._compat import torch_version_le


@pytest.mark.skipif(torch_version_le(2, 0, 0), reason="Autocast not supported")
class TestDeDoDe:
    @pytest.mark.slow
    @pytest.mark.parametrize("descriptor_model", ["B", "G"])
    @pytest.mark.parametrize("detector_model", ["L"])
    def test_smoke(self, dtype, device, descriptor_model, detector_model):
        if "G" in descriptor_model and device.type != "cuda" and dtype == torch.float16:
            pytest.skip('G descriptors do not support no cuda device. "LayerNormKernelImpl" not implemented for `Half`')
        dedode = DeDoDe(descriptor_model=descriptor_model, detector_model=detector_model, amp_dtype=dtype).to(
            device, dtype
        )
        shape = (2, 3, 128, 128)
        n = 1000
        inp = torch.randn(*shape, device=device, dtype=dtype)
        keypoints, scores, descriptions = dedode(inp, n=n)
        assert keypoints.shape == (shape[0], n, 2)
        assert scores.shape == (shape[0], n)
        assert descriptions.shape == (shape[0], n, 256)

    @pytest.mark.slow
    @pytest.mark.parametrize("descriptor_model", ["B", "G"])
    @pytest.mark.parametrize("detector_model", ["L"])
    def test_smoke_amp_fp16(self, dtype, device, descriptor_model, detector_model):
        if "G" in descriptor_model and device.type != "cuda":
            pytest.skip('G descriptors do not support no cuda device. "LayerNormKernelImpl" not implemented for `Half`')
        dedode = DeDoDe(descriptor_model=descriptor_model, detector_model=detector_model, amp_dtype=torch.float16).to(
            device, dtype
        )
        shape = (2, 3, 128, 128)
        n = 1000
        inp = torch.randn(*shape, device=device, dtype=dtype)
        keypoints, scores, descriptions = dedode(inp, n=n)
        assert keypoints.shape == (shape[0], n, 2)
        assert scores.shape == (shape[0], n)
        assert descriptions.shape == (shape[0], n, 256)

    @pytest.mark.slow
    @pytest.mark.parametrize("detector_weights", ["L-upright", "L-C4", "L-SO2", "L-C4-v2"])
    @pytest.mark.parametrize("descriptor_weights", ["B-upright", "B-C4", "B-SO2", "G-upright", "G-C4"])
    def test_pretrained(self, dtype, device, descriptor_weights, detector_weights):
        if "G" in descriptor_weights and device.type != "cuda" and dtype == torch.float16:
            pytest.skip('G descriptors do not support no cuda device. "LayerNormKernelImpl" not implemented for `Half`')
        dedode = DeDoDe.from_pretrained(
            detector_weights=detector_weights,
            descriptor_weights=descriptor_weights,
            amp_dtype=dtype,
        ).to(device, dtype)
        assert isinstance(dedode, DeDoDe)

        shape = (2, 3, 128, 128)
        n = 1000
        inp = torch.randn(*shape, device=device, dtype=dtype)
        keypoints, scores, descriptions = dedode(inp, n=n)

        assert keypoints.shape == (shape[0], n, 2)
        assert scores.shape == (shape[0], n)
        assert descriptions.shape == (shape[0], n, 256)
