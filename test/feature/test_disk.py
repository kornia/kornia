import sys

import pytest
import torch

import kornia.testing as utils  # test utils
from kornia.feature.disk import DISK, DISKFeatures
from kornia.testing import assert_close
from kornia.utils._compat import torch_version_ge


class TestDisk:
    def test_smoke(self, dtype, device):
        disk = DISK().to(device, dtype)
        inp = torch.ones(1, 3, 64, 64, device=device, dtype=dtype)
        output = disk(inp)
        assert isinstance(output, list)
        assert len(output) == 1
        assert all(isinstance(e, DISKFeatures) for e in output)

    def test_smoke_n_detections(self, dtype, device):
        """Unless we give it an actual image and use pretrained weights, we can't expect the number of detections
        to really match the limit."""
        disk = DISK().to(device, dtype)
        inp = torch.ones(1, 3, 64, 64, device=device, dtype=dtype)
        output = disk(inp, n=100)
        assert isinstance(output, list)
        assert len(output) == 1
        assert all(isinstance(e, DISKFeatures) for e in output)

    def test_smoke_pretrained(self, device):
        disk = DISK.from_pretrained(checkpoint='depth', device=device)
        inp = torch.ones(1, 3, 64, 64, device=device)
        output = disk(inp)
        assert isinstance(output, list)
        assert len(output) == 1
        assert all(isinstance(e, DISKFeatures) for e in output)

    @pytest.mark.skipif(sys.platform == "win32", reason="this test takes so much memory in the CI with Windows")
    @pytest.mark.parametrize("data", ["disk_outdoor"], indirect=True)
    def test_pretrained_outdoor(self, device, dtype, data):
        disk = DISK.from_pretrained(checkpoint='depth', device=device).to(dtype)
        data_dev = utils.dict_to(data, device, dtype)
        num_feat = 256
        with torch.no_grad():
            out = disk(data_dev['img1'], num_feat)
        assert_close(out[0].keypoints, data_dev["disk1"][0].keypoints)
        assert_close(out[0].descriptors, data_dev["disk1"][0].descriptors)

    def test_heatmap_and_dense_descriptors(self, dtype, device):
        disk = DISK().to(device, dtype)
        inp = torch.ones(1, 3, 64, 64, device=device, dtype=dtype)
        heatmaps, descriptors = disk.heatmap_and_dense_descriptors(inp)

        assert heatmaps.shape == (1, 1, 64, 64)
        assert descriptors.shape == (1, 128, 64, 64)
        assert heatmaps.dtype == dtype
        assert descriptors.dtype == dtype

    def test_not_divisible_by_16(self, device):
        disk = DISK().to(device)
        inp = torch.ones(1, 3, 72, 64, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)
        
        _ = disk(inp, pad_if_not_divisible=True)

        inp = torch.ones(1, 3, 64, 72, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)
        
        _ = disk(inp, pad_if_not_divisible=True)

        inp = torch.ones(1, 3, 72, 72, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)
        
        _ = disk(inp, pad_if_not_divisible=True)

    def test_wrong_n_channels(self, device):
        disk = DISK().to(device)
        inp = torch.ones(1, 1, 64, 64, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)
