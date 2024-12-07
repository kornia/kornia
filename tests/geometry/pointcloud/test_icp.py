import torch
import kornia

import pypose as pp

class TestICP:
    def test_vanila_icp(self):
        src = torch.randn(2, 3)
        dst = torch.randn(2, 3)

        kornia_results = kornia.utils.pointcloud_io.iterative_closest_point(src, dst)

        icp = pp.module.ICP()
        pp_icp = icp(src, dst)

        torch.allclose(kornia_results, pp_icp)