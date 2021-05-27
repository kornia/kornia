import pytest
import torch

from torch.autograd import gradcheck
from torch.testing import assert_allclose

from kornia.geometry.calibration.undistort import undistort_points


class TestUndistortion:
    def test_smoke(self, device, dtype):
        points = torch.rand(1, 2, device=device, dtype=dtype)
        K = torch.rand(3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(4, device=device, dtype=dtype)

        pointsu = undistort_points(points, K, distCoeff)
        assert points.shape == pointsu.shape

    @pytest.mark.parametrize("num_points, num_distcoeff", [(3,4), (4,5), (5,8), (6,12), (7,14)])
    def test_shape(self, num_points, num_distcoeff, device, dtype):
        N, Ndist = num_points, num_distcoeff

        points = torch.rand(N, 2, device=device, dtype=dtype)
        K = torch.rand(3, 3, device=device, dtype=dtype)
        distCoeff = torch.rand(Ndist, device=device, dtype=dtype)

        pointsu = undistort_points(points, K, distCoeff)
        assert points.shape == (N, 2)
    
    def test_opencv(self, device, dtype):
        pts = torch.tensor([[1028.0374, 788.7520],
                            [1025.1218, 716.8726],
                            [1022.1792, 645.1857]], device=device, dtype=dtype)

        K = torch.tensor([[1.7315e+03, 0.0000e+00, 6.2289e+02],
                          [0.0000e+00, 1.7320e+03, 5.3537e+02],
                          [0.0000e+00, 0.0000e+00, 1.0000e+00]], device=device, dtype=dtype)
        


        # ----- Test 1: using 5 distortion coefficients
        dist1 = torch.tensor([[-0.1007, 0.2650, -0.0018, 0.0007, -0.2597]], device=device, dtype=dtype)
        
        # Expected ouput generated with OpenCV:
        # import cv2
        # ptsu_expected = cv2.undistortPoints(pts.numpy().reshape(-1,1,2), K.numpy(), 
        #                               dist1.numpy(), None, None, K.numpy()).reshape(-1,2)
        ptsu_expected = torch.tensor([[1030.5992, 790.65533],
                                      [1027.3059, 718.10020],
                                      [1024.0700, 645.90600]], device=device, dtype=dtype)
        
        ptsu = undistort_points(pts, K, dist1)
        assert_allclose(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)



        # ----- Test 2: using 14 distortion coefficients
        dist2 = torch.tensor([[-5.6388e-02,  2.3881e-01,  8.3374e-02,  2.0710e-03,  7.1349e+00,
                                5.6335e-02, -3.1738e-01,  4.9981e+00, -4.0287e-03, -2.8246e-02,
                               -8.6064e-02,  1.5543e-02, -1.7322e-01,  2.3154e-03]], device=device, dtype=dtype)
        
        # Expected ouput generated with OpenCV:
        # import cv2
        # ptsu_expected = cv2.undistortPoints(pts.numpy().reshape(-1,1,2), K.numpy(), 
        #                               dist2.numpy(), None, None, K.numpy()).reshape(-1,2)
        ptsu_expected = torch.tensor([[1030.8245, 786.3807],
                                      [1027.5505, 715.0732],
                                      [1024.2753, 644.0319]], device=device, dtype=dtype)
        
        ptsu = undistort_points(pts, K, dist2)
        assert_allclose(ptsu, ptsu_expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device):
        points = torch.rand(8, 2, device=device, dtype=torch.float64, requires_grad=True)
        K = torch.rand(3, 3, device=device, dtype=torch.float64)
        distCoeff = torch.rand(14, device=device, dtype=torch.float64)

        assert gradcheck(undistort_points,
                         (points, K, distCoeff,), raise_exception=True)