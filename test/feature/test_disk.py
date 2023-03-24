import pytest
import torch

from kornia.feature.disk import DISK, DISKFeatures


class TestDisk:
    def test_smoke(self, dtype, device):
        disk = DISK().to(device, dtype)
        inp = torch.ones(1, 3, 64, 64, device=device, dtype=dtype)
        output = disk(inp)
        assert isinstance(output, list)
        assert len(output) == 1
        for element in output:
            assert isinstance(element, DISKFeatures)

    def test_smoke_n_detections(self, dtype, device):
        """Unless we give it an actual image and use pretrained weights, we can't expect the number of detections
        to really match the limit."""
        disk = DISK().to(device, dtype)
        inp = torch.ones(1, 3, 64, 64, device=device, dtype=dtype)
        output = disk(inp, n=100)
        assert isinstance(output, list)
        assert len(output) == 1
        for element in output:
            assert isinstance(element, DISKFeatures)

    def test_smoke_pretrained(self, device):
        disk = DISK.from_pretrained(checkpoint='depth', device=device)
        inp = torch.ones(1, 3, 64, 64, device=device)
        output = disk(inp)
        assert isinstance(output, list)
        assert len(output) == 1
        for element in output:
            assert isinstance(element, DISKFeatures)

    def test_heatmap_and_dense_descriptors(self, dtype, device):
        disk = DISK().to(device, dtype)
        inp = torch.ones(1, 3, 64, 64, device=device, dtype=dtype)
        heatmaps, descriptors = disk.heatmap_and_dense_descriptors(inp)

        assert heatmaps.shape == (1, 1, 64, 64)
        assert descriptors.shape == (1, 128, 64, 64)
        assert heatmaps.dtype == dtype
        assert descriptors.dtype == dtype

    def test_not_divisible_by_16(self, device):
        """This is to be removed when we add automatic padding."""
        disk = DISK().to(device)
        inp = torch.ones(1, 3, 72, 64, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)

        inp = torch.ones(1, 3, 64, 72, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)

        inp = torch.ones(1, 3, 72, 72, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)

    def test_wrong_n_channels(self, device):
        """This is to be removed when we add automatic padding."""
        disk = DISK().to(device)
        inp = torch.ones(1, 1, 64, 64, device=device)
        with pytest.raises(ValueError):
            _ = disk(inp)
