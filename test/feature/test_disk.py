import torch

from kornia.feature.disk import Disk

class TestDisk:
    def test_smoke(self, device):
        print(Disk)
        disk = Disk(pretrained='depth', device=device)
        disk.eval()
        inp = torch.ones(1, 3, 256, 256, device=device)
        output = disk.detect(inp)
        assert output is not None