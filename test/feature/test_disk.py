import torch

from kornia.feature.disk import DISK, DISKFeatures


class TestDisk:
    def test_smoke(self, device):
        disk = DISK().to(device)
        inp = torch.ones(1, 3, 256, 256, device=device)
        output = disk(inp)
        assert isinstance(output, list)
        assert len(output) == 1
        for element in output:
            assert isinstance(element, DISKFeatures)

    def test_smoke_pretrained(self, device):
        disk = DISK.from_pretrained(checkpoint='depth', device=device)
        inp = torch.ones(1, 3, 256, 256, device=device)
        output = disk(inp)
        assert isinstance(output, list)
        assert len(output) == 1
        for element in output:
            assert isinstance(element, DISKFeatures)

    def test_heatmap_and_dense_descriptors(self, device):
        disk = DISK().to(device)
        inp = torch.ones(1, 3, 256, 256, device=device)
        heatmaps, descriptors = disk.heatmap_and_dense_descriptors(inp)

        assert heatmaps.shape == (1, 1, 256, 256)
        assert descriptors.shape == (1, 128, 256, 256)
