import pytest

import torch
import torchgeometry as tgm
from torch.autograd import gradcheck

import utils


class TestSoftArgmax2d:
    def _test_smoke(self):
        input = torch.zeros(1, 1, 2, 3)
        m = tgm.contrib.SpatialSoftArgmax2d()
        assert m(input).shape == (1, 1, 2)

    def _test_top_left(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., 0, 0] = 10.

        coord = tgm.contrib.spatial_soft_argmax2d(input)
        assert pytest.approx(coord[..., 0].item(), -1.0)
        assert pytest.approx(coord[..., 1].item(), -1.0)

    def _test_bottom_right(self):
        input = torch.zeros(1, 1, 2, 3)
        input[..., -1, 1] = 10.

        coord = tgm.contrib.spatial_soft_argmax2d(input)
        assert pytest.approx(coord[..., 0].item(), 1.0)
        assert pytest.approx(coord[..., 1].item(), 1.0)

    def _test_batch2_n2(self):
        input = torch.zeros(2, 2, 2, 3)
        input[0, 0, 0, 0] = 10.  # top-left
        input[0, 1, 0, -1] = 10.  # top-right
        input[1, 0, -1, 0] = 10.  # bottom-left
        input[1, 1, -1, -1] = 10.  # bottom-right

        coord = tgm.contrib.spatial_soft_argmax2d(input)
        assert pytest.approx(coord[0, 0, 0].item(), -1.0)  # top-left
        assert pytest.approx(coord[0, 0, 1].item(), -1.0)
        assert pytest.approx(coord[0, 1, 0].item(), 1.0)  # top-right
        assert pytest.approx(coord[0, 1, 1].item(), -1.0)
        assert pytest.approx(coord[1, 0, 0].item(), -1.0)  # bottom-left
        assert pytest.approx(coord[1, 0, 1].item(), 1.0)
        assert pytest.approx(coord[1, 1, 0].item(), 1.0)  # bottom-right
        assert pytest.approx(coord[1, 1, 1].item(), 1.0)

    # TODO: implement me
    def _test_jit(self):
        pass

    def _test_gradcheck(self):
        input = torch.rand(2, 3, 3, 2)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(tgm.contrib.spatial_soft_argmax2d,
                         (input), raise_exception=True)

    def test_run_all(self):
        self._test_smoke()
        self._test_top_left()
        self._test_bottom_right()
        self._test_batch2_n2()
        self._test_gradcheck()
