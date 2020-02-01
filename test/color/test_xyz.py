import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose
from test.common import device


class TestRgbToXyz:

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    def test_rgb_to_xyz(self, batch_size, device):

        data = torch.tensor([[[0.9637, 0.0586, 0.6470, 0.6212, 0.9622],
                              [0.8293, 0.4858, 0.8953, 0.2607, 0.3250],
                              [0.5314, 0.4189, 0.8388, 0.8065, 0.2211],
                              [0.9682, 0.2928, 0.4118, 0.2533, 0.0455]],

                             [[0.6936, 0.3457, 0.9466, 0.9937, 0.2692],
                              [0.7485, 0.7320, 0.8323, 0.6889, 0.4831],
                              [0.1865, 0.7439, 0.1366, 0.8858, 0.2077],
                              [0.6227, 0.6140, 0.3936, 0.5024, 0.4157]],

                             [[0.6477, 0.9269, 0.7531, 0.7349, 0.9485],
                                 [0.4264, 0.8539, 0.9830, 0.2269, 0.1138],
                                 [0.3988, 0.1605, 0.6220, 0.0546, 0.1106],
                                 [0.2128, 0.5673, 0.0781, 0.1431, 0.3310]]])

        # Reference output generated using OpenCV: cv2.cvtColor(data, cv2.COLOR_RGB2XYZ)
        xyz_ref = torch.tensor([[[0.7623584, 0.31501925, 0.7412189, 0.7441359, 0.66425407],
                                 [0.6866283, 0.61618143, 0.84423876, 0.39480132, 0.32732624],
                                 [0.3578189, 0.4677382, 0.50703406, 0.6592388, 0.18541752],
                                 [0.6603961, 0.44267434, 0.32468265, 0.30994105, 0.22713262]],

                                [[0.7477299, 0.32658678, 0.86891913, 0.89580274, 0.4656054],
                                 [0.7424382, 0.6884378, 0.8565741, 0.5644922, 0.4228247],
                                 [0.2751717, 0.63267857, 0.3209684, 0.8089483, 0.20354219],
                                 [0.6665957, 0.5423198, 0.37470126, 0.42349333, 0.33085644]],

                                [[0.7167665, 0.92310345, 0.8409531, 0.82877415, 0.95198023],
                                 [0.51042646, 0.9080406, 1.0505873, 0.30275896, 0.17200153],
                                 [0.4114541, 0.24927814, 0.62354034, 0.17305644, 0.13412625],
                                 [0.29514894, 0.6179093, 0.12908883, 0.20075734, 0.3649534]]])

        data.to(device)
        xyz_ref.to(device)

        if batch_size is not None:
            data = data.repeat(batch_size, 1, 1, 1)
            xyz_ref = xyz_ref.repeat(batch_size, 1, 1, 1)

        xyz = kornia.color.RgbToXyz()
        out = xyz(data)
        assert_allclose(out, xyz_ref)

    def test_grad(self, device):

        data = torch.rand(2, 3, 4, 5).to(device)
        data = utils.tensor_to_gradcheck_var(data)
        assert gradcheck(kornia.color.RgbToXyz(), (data,), raise_exception=True)

    @pytest.mark.parametrize("input_shape", [(2, 2), (3, 3, 5, 3, 3)])
    def test_shape(self, input_shape, device):
        with pytest.raises(ValueError):
            xyz = kornia.color.RgbToXyz()
            out = xyz(torch.ones(*input_shape).to(device))

    def test_inverse(self, device):
        data = torch.rand(3, 4, 5).to(device)
        xyz = kornia.color.XyzToRgb()
        rgb = kornia.color.RgbToXyz()

        data_out = xyz(rgb(data))
        assert_allclose(data_out, data)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):

        data = torch.rand((2, 3, 4, 5)).to(device)
        xyz = kornia.color.RgbToXyz()
        xyz_traced = torch.jit.trace(kornia.color.RgbToXyz(), data)
        assert_allclose(xyz_traced(data), xyz(data))


class TestXyzToRgb:

    @pytest.mark.parametrize("batch_size", [None, 1, 2, 5])
    def test_xyz_to_rgb(self, batch_size, device):

        data = torch.tensor([[[0.6315228, 0.4196397, 0.33123854, 0.3277169, 0.20501322],
                              [0.44133404, 0.4823162, 0.528561, 0.51644784, 0.28001237],
                              [0.31131047, 0.8453884, 0.4486181, 0.6015828, 0.42048606],
                              [0.5472367, 0.48154795, 0.36668795, 0.39913517, 0.40271503]],

                             [[0.79137707, 0.501063, 0.3700857, 0.57410157, 0.15295872],
                              [0.570678, 0.76664513, 0.48567873, 0.47680324, 0.2583247],
                              [0.38080955, 0.9315215, 0.4404478, 0.50659215, 0.5984908],
                              [0.5388581, 0.76993656, 0.4027568, 0.5952581, 0.68663263]],

                             [[0.86013114, 0.17629854, 0.83010703, 0.27881518, 0.30543375],
                                 [0.17009716, 0.61201245, 0.33521807, 0.15526368, 0.7401195],
                                 [0.34011865, 0.6541383, 0.96909684, 0.43090558, 0.70467836],
                                 [0.6738866, 0.47461915, 0.91508406, 0.44147202, 0.14099535]]])
        # Reference output generated using OpenCV: cv2.cvtColor(data, cv2.COLOR_XYZ2RGB)
        rgb_ref = torch.tensor([[[0.4011656, 0.5017336, 0.09065688, 0.04048038, 0.2769511],
                                 [0.46811658, 0.07937729, 0.79911184, 0.8632158, 0.1413149],
                                 [0.2538725, 0.98146427, 0.29357457, 0.95588684, 0.09129933],
                                 [0.6090472, 0.14032376, 0.11294556, 0.15829873, 0.1792411]],

                                [[0.90825266, 0.54057753, 0.40771842, 0.7709542, 0.1009315],
                                 [0.64988965, 0.99616426, 0.41274834, 0.40036052, 0.24396756],
                                 [0.42678973, 0.95531154, 0.43172216, 0.3851813, 0.7444883],
                                 [0.5084846, 0.99737406, 0.4381809, 0.7481805, 0.9036418]],

                                [[0.78309417, 0.10751611, 0.82060075, 0.19588977, 0.3031369],
                                 [0.08796211, 0.51749885, 0.28474382, 0.09561294, 0.74540937],
                                 [0.2992335, 0.5486014, 0.95973116, 0.38571155, 0.64634556],
                                 [0.6330101, 0.3715171, 0.90575427, 0.36752605, 0.03138365]]])

        data.to(device)
        rgb_ref.to(device)

        if batch_size is not None:
            data = data.repeat(batch_size, 1, 1, 1)
            rgb_ref = rgb_ref.repeat(batch_size, 1, 1, 1)

        rgb = kornia.color.XyzToRgb()
        rgb_out = rgb(data)

        assert_allclose(rgb_out, rgb_ref)

    def test_grad(self, device):

        data = torch.rand(2, 3, 4, 5).to(device)
        data = utils.tensor_to_gradcheck_var(data)
        assert gradcheck(kornia.color.XyzToRgb(), (data,), raise_exception=True)

    def test_inverse(self, device):
        data = torch.rand(3, 4, 5).to(device)
        xyz = kornia.color.XyzToRgb()
        rgb = kornia.color.RgbToXyz()

        data_out = rgb(xyz(data))
        assert_allclose(data_out, data)

    @pytest.mark.parametrize("input_shape", [([2, 2],), ([3, 3, 7, 3, 3],)])
    def test_shape(self, input_shape, device):
        with pytest.raises(ValueError):
            rgb = kornia.color.XyzToRgb()
            out = rgb(torch.ones(*input_shape).to(device))

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device):

        data = torch.rand((2, 3, 4, 5)).to(device)
        xyz = kornia.color.XyzToRgb()
        xyz_traced = torch.jit.trace(kornia.color.XyzToRgb(), data)
        assert_allclose(xyz_traced(data), xyz(data))
