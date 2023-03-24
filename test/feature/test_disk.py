import pytest
import torch

from kornia.feature.disk import DISK, DISKFeatures


class TestDisk:
    def test_smoke(self, device):
        disk = DISK().to(device)
        inp = torch.ones(1, 3, 64, 64, device=device)
        output = disk(inp)
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

    def test_heatmap_and_dense_descriptors(self, device):
        disk = DISK().to(device)
        inp = torch.ones(1, 3, 64, 64, device=device)
        heatmaps, descriptors = disk.heatmap_and_dense_descriptors(inp)

        assert heatmaps.shape == (1, 1, 64, 64)
        assert descriptors.shape == (1, 128, 64, 64)

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
