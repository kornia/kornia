import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature import LoFTR
from kornia.geometry import resize
from kornia.testing import assert_close


class TestLoFTR:
    @pytest.mark.skipif(torch.__version__.startswith('1.6'),
                        reason='1.6.0 not supporting the pretrained weights as they are packed.')
    def test_pretrained_outdoor_smoke(self, device):
        if device == torch.device('cpu'):
            loftr = LoFTR('outdoor').to(device)

    @pytest.mark.skipif(torch.__version__.startswith('1.6'),
                        reason='1.6.0 not supporting the pretrained weights as they are packed.')
    def test_pretrained_indoor_smoke(self, device):
        if device == torch.device('cpu'):
            loftr = LoFTR('indoor').to(device)

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
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 1, 1, 32, 32
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        patches2x = resize(patches, (48, 48))
        input = {"image0": patches, "image1": patches2x}
        model = LoFTR().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(LoFTR().to(patches.device, patches.dtype).eval())
        out = model(input)
        out_jit = model(input)
        for k, v in out.items():
            assert_close(v, out_jit[k])
