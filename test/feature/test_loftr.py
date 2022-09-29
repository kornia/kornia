import sys

import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature import LoFTR
from kornia.geometry import resize
from kornia.testing import assert_close
from kornia.utils._compat import torch_version_geq


class TestLoFTR:
    def test_pretrained_outdoor_smoke(self, device, dtype):
        loftr = LoFTR('outdoor').to(device, dtype)
        assert loftr is not None

    def test_pretrained_indoor_smoke(self, device, dtype):
        loftr = LoFTR('indoor').to(device, dtype)
        assert loftr is not None

    @pytest.mark.skipif(torch_version_geq(1, 10), reason="RuntimeError: CUDA out of memory with pytorch>=1.10")
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_fund"], indirect=True)
    def test_pretrained_indoor(self, device, dtype, data):
        loftr = LoFTR('indoor').to(device, dtype)
        data_dev = utils.dict_to(data, device, dtype)
        with torch.no_grad():
            out = loftr(data_dev)
        assert_close(out['keypoints0'], data_dev["loftr_indoor_tentatives0"])
        assert_close(out['keypoints1'], data_dev["loftr_indoor_tentatives1"])

    @pytest.mark.skipif(torch_version_geq(1, 10), reason="RuntimeError: CUDA out of memory with pytorch>=1.10")
    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["loftr_homo"], indirect=True)
    def test_pretrained_outdoor(self, device, dtype, data):
        loftr = LoFTR('outdoor').to(device, dtype)
        data_dev = utils.dict_to(data, device, dtype)
        with torch.no_grad():
            out = loftr(data_dev)
        assert_close(out['keypoints0'], data_dev["loftr_outdoor_tentatives0"])
        assert_close(out['keypoints1'], data_dev["loftr_outdoor_tentatives1"])

    def test_mask(self, device):
        patches = torch.rand(1, 1, 32, 32, device=device)
        mask = torch.rand(1, 32, 32, device=device)
        loftr = LoFTR().to(patches.device, patches.dtype)
        input = {"image0": patches, "image1": patches, "mask0": mask, "mask1": mask}
        with torch.no_grad():
            out = loftr(input)
        assert out is not None

    @pytest.mark.skip("Takes too long time (but works)")
    def test_gradcheck(self, device):
        patches = torch.rand(1, 1, 32, 32, device=device)
        patches05 = resize(patches, (48, 48))
        patches = utils.tensor_to_gradcheck_var(patches)  # to var
        patches05 = utils.tensor_to_gradcheck_var(patches05)  # to var
        loftr = LoFTR().to(patches.device, patches.dtype)

        def proxy_forward(x, y):
            return loftr.forward({"image0": x, "image1": y})["keypoints0"]

        assert gradcheck(proxy_forward, (patches, patches05), eps=1e-4, atol=1e-4, raise_exception=True)

    @pytest.mark.skip("does not like transformer.py:L99, zip iteration")
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        patches2x = resize(patches, (48, 48))
        input = {"image0": patches, "image1": patches2x}
        model = LoFTR().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        out = model(input)
        out_jit = model_jit(input)
        for k, v in out.items():
            assert_close(v, out_jit[k])
