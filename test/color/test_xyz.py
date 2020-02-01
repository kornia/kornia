import pytest

import kornia
import kornia.testing as utils  # test utils

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

class TestRgbToXyz:

    def test_rgb_to_xyz(self, device):
        data =   torch.tensor([[[0.9637, 0.0586, 0.6470, 0.6212, 0.9622],
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
        x_ref = torch.tensor([[0.7623584 , 0.31501925, 0.7412189 , 0.7441359 , 0.66425407],
                                [0.6866283 , 0.61618143, 0.84423876, 0.39480132, 0.32732624],
                                [0.3578189 , 0.4677382 , 0.50703406, 0.6592388 , 0.18541752],
                                [0.6603961 , 0.44267434, 0.32468265, 0.30994105, 0.22713262]])

        y_ref = torch.tensor([[0.7477299 , 0.32658678, 0.86891913, 0.89580274, 0.4656054 ],
                                [0.7424382 , 0.6884378 , 0.8565741 , 0.5644922 , 0.4228247 ],
                                [0.2751717 , 0.63267857, 0.3209684 , 0.8089483 , 0.20354219],
                                [0.6665957 , 0.5423198 , 0.37470126, 0.42349333, 0.33085644]])
        z_ref = torch.tensor([[0.7167665 , 0.92310345, 0.8409531 , 0.82877415, 0.95198023],
                                [0.51042646, 0.9080406 , 1.0505873 , 0.30275896, 0.17200153],
                                [0.4114541 , 0.24927814, 0.62354034, 0.17305644, 0.13412625],
                                [0.29514894, 0.6179093 , 0.12908883, 0.20075734, 0.3649534 ]])
        xyz_ref = torch.stack((x_ref, y_ref, z_ref), dim=-3)                         
        
        
        data.to(device)
        xyz_ref.to(device)

        xyz = kornia.color.RgbToXyz()
        out = xyz(data)
        assert_allclose(out, xyz_ref)

    def test_grad(self, device):
        data = torch.rand(2,3,4,5).to(device)
        data = utils.tensor_to_gradcheck_var(data)
        assert gradcheck(kornia.rgb_to_xyz, (data,), raise_exception=True)


        
