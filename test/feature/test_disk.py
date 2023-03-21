import torch

from kornia.feature.disk import DISK

class TestDisk:
    def test_smoke_depth(self, device):
        disk: DISK = DISK.from_pretrained(checkpoint='depth', device=device)
        inp = torch.ones(1, 3, 256, 256, device=device)
        output = disk.detect(inp)
        assert output is not None

    def test_smoke_epipolar(self, device):
        disk: DISK = DISK.from_pretrained(checkpoint='epipolar', device=device)
        inp = torch.ones(1, 3, 256, 256, device=device)
        output = disk.detect(inp)
        assert output is not None

    def test_heatmap_and_dense_descriptors(self, device):
        disk: DISK = DISK.from_pretrained(checkpoint='depth', device=device)
        inp = torch.ones(1, 3, 256, 256, device=device)
        heatmaps, descriptors = disk.heatmap_and_dense_descriptors(inp)

        assert heatmaps.shape == (1, 1, 256, 256)
        assert descriptors.shape == (1, 128, 256, 256)