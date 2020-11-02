import pytest
import torch
from torch.testing import assert_allclose

from kornia.augmentation.random_generator import (
    random_rotation_generator3d,
    random_affine_generator3d,
    random_motion_blur_generator3d,
    center_crop_generator3d,
    random_crop_generator3d,
    random_perspective_generator3d,
)


class RandomGeneratorBaseTests():

    def test_valid_param_combinations(self, device, dtype):
        raise NotImplementedError

    def test_invalid_param_combinations(self, device, dtype):
        raise NotImplementedError

    def test_random_gen(self, device, dtype):
        raise NotImplementedError

    def test_same_on_batch(self, device, dtype):
        raise NotImplementedError


class TestRandomPerspectiveGen3D(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('depth,height,width', [(200, 200, 200)])
    @pytest.mark.parametrize('distortion_scale', [torch.tensor(0.), torch.tensor(0.5), torch.tensor(1.)])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, depth, height, width, distortion_scale, batch_size, same_on_batch, device, dtype
    ):
        random_perspective_generator3d(
            batch_size=batch_size, depth=depth, height=height, width=width,
            distortion_scale=distortion_scale.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    @pytest.mark.parametrize('depth,height,width,distortion_scale', [
        # Should be failed if distortion_scale > 1. or distortion_scale < 0.
        pytest.param(-100, 100, 100, torch.tensor(0.5), marks=pytest.mark.xfail),
        pytest.param(100, -100, 100, torch.tensor(0.5), marks=pytest.mark.xfail),
        pytest.param(100, 100, -100, torch.tensor(-0.5), marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor(1.5), marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor([0., 0.5]), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, depth, height, width, distortion_scale, device, dtype):
        random_perspective_generator3d(
            batch_size=8, height=height, width=width,
            distortion_scale=distortion_scale.to(device=device, dtype=dtype))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        res = random_perspective_generator3d(
            batch_size, 200, 200, 200, torch.tensor(0.5, device=device, dtype=dtype))
        expected = dict(
            start_points=torch.tensor([[[0., 0., 0.],
                                        [199., 0., 0.],
                                        [199., 199., 0.],
                                        [0., 199., 0.],
                                        [0., 0., 199.],
                                        [199., 0., 199.],
                                        [199., 199., 199.],
                                        [0., 199., 199.]],
                                       [[0., 0., 0.],
                                        [199., 0., 0.],
                                        [199., 199., 0.],
                                        [0., 199., 0.],
                                        [0., 0., 199.],
                                        [199., 0., 199.],
                                        [199., 199., 199.],
                                        [0., 199., 199.]]], device=device, dtype=dtype),
            end_points=torch.tensor([[[2.9077, 3.1455, 6.1793],
                                     [196.3710, 26.3086, 23.8392],
                                     [151.2382, 152.5624, 4.1772],
                                     [6.6320, 191.1473, 18.7684],
                                     [42.1260, 43.5251, 180.1234],
                                     [168.3782, 4.4055, 163.9409],
                                     [167.8298, 177.1361, 195.2633],
                                     [34.1715, 183.3881, 183.5471]],
                                    [[1.5670, 2.0183, 46.5954],
                                     [191.3937, 13.2522, 6.5217],
                                     [186.4072, 187.3296, 11.2833],
                                     [15.8784, 168.2796, 29.8886],
                                     [42.4501, 1.6248, 183.5378],
                                     [177.3848, 46.7149, 192.8844],
                                     [177.6152, 184.5908, 149.9299],
                                     [42.2815, 179.7861, 194.0347]]], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['start_points'], expected['start_points'], atol=1e-4, rtol=1e-4)
        assert_allclose(res['end_points'], expected['end_points'], atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        res = random_perspective_generator3d(
            batch_size, 200, 200, 200, torch.tensor(0.5, device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            start_points=torch.tensor([[[0., 0., 0.],
                                        [199., 0., 0.],
                                        [199., 199., 0.],
                                        [0., 199., 0.],
                                        [0., 0., 199.],
                                        [199., 0., 199.],
                                        [199., 199., 199.],
                                        [0., 199., 199.]],
                                       [[0., 0., 0.],
                                        [199., 0., 0.],
                                        [199., 199., 0.],
                                        [0., 199., 0.],
                                        [0., 0., 199.],
                                        [199., 0., 199.],
                                        [199., 199., 199.],
                                        [0., 199., 199.]]], device=device, dtype=dtype),
            end_points=torch.tensor([[[2.9077, 3.1455, 6.1793],
                                      [196.3710, 26.3086, 23.8392],
                                      [151.2382, 152.5624, 4.1772],
                                      [6.6320, 191.1473, 18.7684],
                                      [42.1260, 43.5251, 180.1234],
                                      [168.3782, 4.4055, 163.9409],
                                      [167.8298, 177.1361, 195.2633],
                                      [34.1715, 183.3881, 183.5471]],
                                     [[2.9077, 3.1455, 6.1793],
                                      [196.3710, 26.3086, 23.8392],
                                      [151.2382, 152.5624, 4.1772],
                                      [6.6320, 191.1473, 18.7684],
                                      [42.1260, 43.5251, 180.1234],
                                      [168.3782, 4.4055, 163.9409],
                                      [167.8298, 177.1361, 195.2633],
                                      [34.1715, 183.3881, 183.5471]]], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['start_points'], expected['start_points'])
        assert_allclose(res['end_points'], expected['end_points'])


class TestRandomAffineGen3D(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('depth,height,width', [(200, 300, 400)])
    @pytest.mark.parametrize('degrees', [torch.tensor([(0, 30), (0, 30), (0, 30)])])
    @pytest.mark.parametrize('translate', [None, torch.tensor([0.1, 0.1, 0.1])])
    @pytest.mark.parametrize('scale', [None, torch.tensor([[0.7, 1.2], [0.7, 1.2], [0.7, 1.2]])])
    @pytest.mark.parametrize('shear', [None, torch.tensor([[0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [0, 20]])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_size, depth, height, width, degrees, translate, scale, shear, same_on_batch, device, dtype
    ):
        random_affine_generator3d(
            batch_size=batch_size, depth=depth, height=height, width=width,
            degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shears=shear.to(device=device, dtype=dtype) if shear is not None else None,
            same_on_batch=same_on_batch)

    @pytest.mark.parametrize('depth,height,width,degrees,translate,scale,shear', [
        pytest.param(-100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(100, -100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, -100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor([0, 9]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     torch.tensor([0.1, 0.2]), None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     torch.tensor([0.1, 0.2]), None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     torch.tensor([0.1]), None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     None, torch.tensor([[0.2, 0.2, 0.2]]), None, marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     None, torch.tensor([0.2]), None, marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     None, None, torch.tensor([[20, 20, 30]]), marks=pytest.mark.xfail),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     None, None, torch.tensor([20]), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(
        self, depth, height, width, degrees, translate, scale, shear, device, dtype
    ):
        batch_size = 8
        random_affine_generator3d(
            batch_size=batch_size, depth=depth, height=height, width=width,
            degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shears=shear.to(device=device, dtype=dtype) if shear is not None else None)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([[10, 20], [10, 20], [10, 20]])
        translate = torch.tensor([0.1, 0.1, 0.1])
        scale = torch.tensor([[0.7, 1.2], [0.7, 1.2], [0.7, 1.2]])
        shear = torch.tensor([[0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [0, 20]])
        res = random_affine_generator3d(
            batch_size=2, depth=200, height=200, width=200,
            degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shears=shear.to(device=device, dtype=dtype) if shear is not None else None)
        expected = dict(
            translations=torch.tensor(
                [[13.7008, -4.8987, -16.4756], [14.8200, 4.4975, 8.0473]], device=device, dtype=dtype),
            center=torch.tensor(
                [[99.5000, 99.5000, 99.5000], [99.5000, 99.5000, 99.5000]], device=device, dtype=dtype),
            scale=torch.tensor(
                [[1.1776, 0.7418, 0.7785], [1.1644, 0.7663, 0.8877]], device=device, dtype=dtype),
            angles=torch.tensor(
                [[10.5815, 11.2359, 15.2617], [10.6291, 10.5258, 14.7678]], device=device, dtype=dtype),
            sxy=torch.tensor([12.4681, 8.7456], device=device, dtype=dtype),
            sxz=torch.tensor([1.4947, 13.6686], device=device, dtype=dtype),
            syx=torch.tensor([6.2448, 6.1812], device=device, dtype=dtype),
            syz=torch.tensor([0.6268, 0.8073], device=device, dtype=dtype),
            szx=torch.tensor([18.6382, 3.0425], device=device, dtype=dtype),
            szy=torch.tensor([5.3009, 2.6087], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['translations'], expected['translations'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['center'], expected['center'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['scale'], expected['scale'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['angles'], expected['angles'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['sxy'], expected['sxy'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['sxz'], expected['sxz'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['syx'], expected['syx'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['syz'], expected['syz'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['szx'], expected['szx'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['szy'], expected['szy'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([[10, 20], [10, 20], [10, 20]])
        translate = torch.tensor([0.1, 0.1, 0.1])
        scale = torch.tensor([[0.7, 1.2], [0.7, 1.2], [0.7, 1.2]])
        shear = torch.tensor([[0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [0, 20]])
        res = random_affine_generator3d(
            batch_size=2, depth=200, height=200, width=200,
            degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shears=shear.to(device=device, dtype=dtype) if shear is not None else None,
            same_on_batch=True)
        expected = dict(
            translations=torch.tensor(
                [[18.2094, 17.1501, -16.6583], [18.2094, 17.1501, -16.6583]], device=device, dtype=dtype),
            center=torch.tensor(
                [[99.5000, 99.5000, 99.5000], [99.5000, 99.5000, 99.5000]], device=device, dtype=dtype),
            scale=torch.tensor(
                [[0.7263, 0.9631, 0.9384], [0.7263, 0.9631, 0.9384]], device=device, dtype=dtype),
            angles=torch.tensor(
                [[10.5815, 10.6291, 11.2359], [10.5815, 10.6291, 11.2359]], device=device, dtype=dtype),
            sxy=torch.tensor([2.6528, 2.6528], device=device, dtype=dtype),
            sxz=torch.tensor([3.1411, 3.1411], device=device, dtype=dtype),
            syx=torch.tensor([7.5073, 7.5073], device=device, dtype=dtype),
            syz=torch.tensor([16.8504, 16.8504], device=device, dtype=dtype),
            szx=torch.tensor([17.4100, 17.4100], device=device, dtype=dtype),
            szy=torch.tensor([7.5507, 7.5507], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['translations'], expected['translations'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['center'], expected['center'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['scale'], expected['scale'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['angles'], expected['angles'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['sxy'], expected['sxy'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['sxz'], expected['sxz'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['syx'], expected['syx'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['syz'], expected['syz'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['szx'], expected['szx'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['szy'], expected['szy'], rtol=1e-4, atol=1e-4)


class TestRandomRotationGen3D(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('degrees', [torch.tensor([[0, 30], [0, 30], [0, 30]])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, degrees, same_on_batch, device, dtype):
        random_rotation_generator3d(
            batch_size=batch_size, degrees=degrees.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    @pytest.mark.parametrize('degrees', [
        pytest.param(torch.tensor(10), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([10]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([[0, 30]]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([[0, 30], [0, 30]]), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, degrees, device, dtype):
        batch_size = 8
        random_rotation_generator3d(
            batch_size=batch_size, degrees=degrees.to(device=device, dtype=dtype))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([[0, 30], [0, 30], [0, 30]])
        res = random_rotation_generator3d(
            batch_size=2, degrees=degrees.to(device=device, dtype=dtype), same_on_batch=False)
        expected = dict(
            yaw=torch.tensor([1.7446, 1.8873], device=device, dtype=dtype),
            pitch=torch.tensor([3.7076, 1.5774], device=device, dtype=dtype),
            roll=torch.tensor([15.7852, 14.3035], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['yaw'], expected['yaw'], atol=1e-4, rtol=1e-4)
        assert_allclose(res['pitch'], expected['pitch'], atol=1e-4, rtol=1e-4)
        assert_allclose(res['roll'], expected['roll'], atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([[0, 30], [0, 30], [0, 30]])
        res = random_rotation_generator3d(
            batch_size=2, degrees=degrees.to(device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            yaw=torch.tensor([1.7446, 1.7446], device=device, dtype=dtype),
            pitch=torch.tensor([1.8873, 1.8873], device=device, dtype=dtype),
            roll=torch.tensor([3.7076, 3.7076], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['yaw'], expected['yaw'], atol=1e-4, rtol=1e-4)
        assert_allclose(res['pitch'], expected['pitch'], atol=1e-4, rtol=1e-4)
        assert_allclose(res['roll'], expected['roll'], atol=1e-4, rtol=1e-4)


# class TestRandomCropGen(RandomGeneratorBaseTests):

#     @pytest.mark.parametrize('batch_size', [2])
#     @pytest.mark.parametrize('input_size', [(200, 200)])
#     @pytest.mark.parametrize('size', [(100, 100), torch.tensor([[50, 50], [60, 60]])])
#     @pytest.mark.parametrize('resize_to', [None, (100, 100)])
#     @pytest.mark.parametrize('same_on_batch', [True, False])
#     def test_valid_param_combinations(
#         self, batch_size, input_size, size, resize_to, same_on_batch, device, dtype
#     ):
#         random_crop_generator(
#             batch_size=batch_size, input_size=input_size,
#             size=size.to(device=device, dtype=dtype) if isinstance(size, torch.Tensor) else size,
#             resize_to=resize_to,
#             same_on_batch=same_on_batch)

#     @pytest.mark.parametrize('input_size,size,resize_to', [
#         pytest.param((-300, 300), (200, 200), (100, 100), marks=pytest.mark.xfail),
#         pytest.param((100, 100), (200, 200), (100, 100), marks=pytest.mark.xfail),
#         pytest.param((200, 200), torch.tensor([50, 50]), (100, 100), marks=pytest.mark.xfail),
#         pytest.param((100, 100), torch.tensor([[200, 200], [200, 200]]), (100, 100), marks=pytest.mark.xfail),
#     ])
#     def test_invalid_param_combinations(self, input_size, size, resize_to, device, dtype):
#         batch_size = 2
#         random_crop_generator(
#             batch_size=batch_size, input_size=input_size,
#             size=size.to(device=device, dtype=dtype) if isinstance(size, torch.Tensor) else size,
#             resize_to=resize_to)

#     def test_random_gen(self, device, dtype):
#         torch.manual_seed(42)
#         degrees = torch.tensor([10, 20])
#         res = random_crop_generator(
#             batch_size=2, input_size=(100, 100),
#             size=torch.tensor([[50, 60], [70, 80]], device=device, dtype=dtype),
#             resize_to=(200, 200))
#         expected = dict(
#             src=torch.tensor([
#                 [[2, 6],
#                  [61, 6],
#                  [61, 55],
#                  [2, 55]],
#                 [[1, 1],
#                  [80, 1],
#                  [80, 70],
#                  [1, 70]]], device=device, dtype=torch.long),
#             dst=torch.tensor([
#                 [[0, 0],
#                  [199, 0],
#                  [199, 199],
#                  [0, 199]],
#                 [[0, 0],
#                  [199, 0],
#                  [199, 199],
#                  [0, 199]]], device=device, dtype=torch.long),
#         )
#         assert res.keys() == expected.keys()
#         assert_allclose(res['src'], expected['src'])
#         assert_allclose(res['dst'], expected['dst'])

#     def test_same_on_batch(self, device, dtype):
#         torch.manual_seed(42)
#         degrees = torch.tensor([10, 20])
#         res = random_crop_generator(
#             batch_size=2, input_size=(100, 100),
#             size=torch.tensor([[50, 60], [70, 80]], device=device, dtype=dtype),
#             resize_to=(200, 200), same_on_batch=True)
#         expected = dict(
#             src=torch.tensor([
#                 [[2, 3],
#                  [61, 3],
#                  [61, 52],
#                  [2, 52]],
#                 [[2, 3],
#                  [81, 3],
#                  [81, 72],
#                  [2, 72]]], device=device, dtype=torch.long),
#             dst=torch.tensor([
#                 [[0, 0],
#                  [199, 0],
#                  [199, 199],
#                  [0, 199]],
#                 [[0, 0],
#                  [199, 0],
#                  [199, 199],
#                  [0, 199]]], device=device, dtype=torch.long),
#         )
#         assert res.keys() == expected.keys()
#         assert_allclose(res['src'], expected['src'])
#         assert_allclose(res['dst'], expected['dst'])


# class TestCenterCropGen(RandomGeneratorBaseTests):

#     @pytest.mark.parametrize('batch_size', [2])
#     @pytest.mark.parametrize('height', [200])
#     @pytest.mark.parametrize('width', [200])
#     @pytest.mark.parametrize('size', [(100, 100)])
#     def test_valid_param_combinations(
#         self, batch_size, height, width, size, device, dtype
#     ):
#         center_crop_generator(batch_size=batch_size, height=height, width=width, size=size)

#     @pytest.mark.parametrize('height,width,size', [
#         pytest.param(200, -200, (100, 100), marks=pytest.mark.xfail),
#         pytest.param(-200, 200, (100, 100), marks=pytest.mark.xfail),
#         pytest.param(100, 100, (120, 120), marks=pytest.mark.xfail),
#         pytest.param(150, 100, (120, 120), marks=pytest.mark.xfail),
#         pytest.param(100, 150, (120, 120), marks=pytest.mark.xfail),
#     ])
#     def test_invalid_param_combinations(self, height, width, size, device, dtype):
#         batch_size = 2
#         center_crop_generator(batch_size=batch_size, height=height, width=width, size=size)

#     def test_random_gen(self, device, dtype):
#         torch.manual_seed(42)
#         res = center_crop_generator(batch_size=2, height=200, width=200, size=(120, 150))
#         expected = dict(
#             src=torch.tensor([
#                 [[25, 40],
#                  [174, 40],
#                  [174, 159],
#                  [25, 159]],
#                 [[25, 40],
#                  [174, 40],
#                  [174, 159],
#                  [25, 159]]], device=device, dtype=torch.long),
#             dst=torch.tensor([
#                 [[0, 0],
#                  [149, 0],
#                  [149, 119],
#                  [0, 119]],
#                 [[0, 0],
#                  [149, 0],
#                  [149, 119],
#                  [0, 119]]], device=device, dtype=torch.long),
#         )
#         assert res.keys() == expected.keys()
#         assert_allclose(res['src'], expected['src'])
#         assert_allclose(res['dst'], expected['dst'])

#     def test_same_on_batch(self, device, dtype):
#         pass


# class TestRandomMotionBlur(RandomGeneratorBaseTests):

#     @pytest.mark.parametrize('batch_size', [1, 8])
#     @pytest.mark.parametrize('kernel_size', [1, (3, 5)])
#     @pytest.mark.parametrize('angle', [torch.tensor([10, 30])])
#     @pytest.mark.parametrize('direction', [torch.tensor([-1, -1]), torch.tensor([1, 1])])
#     @pytest.mark.parametrize('same_on_batch', [True, False])
#     def test_valid_param_combinations(self, batch_size, kernel_size, angle, direction, same_on_batch, device, dtype):
#         random_motion_blur_generator(
#             batch_size=batch_size, kernel_size=kernel_size,
#             angle=angle.to(device=device, dtype=dtype),
#             direction=direction.to(device=device, dtype=dtype),
#             same_on_batch=same_on_batch)

#     @pytest.mark.parametrize('kernel_size,angle,direction', [
#         pytest.param(4, torch.tensor([30, 100]), torch.tensor([-1, 1]), marks=pytest.mark.xfail),
#         pytest.param(1, torch.tensor([30, 100]), torch.tensor([-1, 1]), marks=pytest.mark.xfail),
#         pytest.param((1, 2, 3), torch.tensor([30, 100]), torch.tensor([-1, 1]), marks=pytest.mark.xfail),
#         pytest.param(3, torch.tensor([30, 100]), torch.tensor([-2, 1]), marks=pytest.mark.xfail),
#         pytest.param(3, torch.tensor([30, 100]), torch.tensor([-1, 2]), marks=pytest.mark.xfail),
#     ])
#     def test_invalid_param_combinations(self, kernel_size, angle, direction, device, dtype):
#         random_motion_blur_generator(
#             batch_size=8, kernel_size=kernel_size, angle=angle.to(device=device, dtype=dtype),
#             direction=direction.to(device=device, dtype=dtype))

#     def test_random_gen(self, device, dtype):
#         torch.manual_seed(42)
#         angle = torch.tensor([30, 90])
#         direction = torch.tensor([-1, 1])
#         res = random_motion_blur_generator(
#             batch_size=2, kernel_size=3, angle=angle.to(device=device, dtype=dtype),
#             direction=direction.to(device=device, dtype=dtype), same_on_batch=False)
#         expected = dict(
#             ksize_factor=torch.tensor([3., 3.], device=device, dtype=dtype),
#             angle_factor=torch.tensor([33.4893, 33.7746], device=device, dtype=dtype),
#             direction_factor=torch.tensor([-0.7528, -0.8948], device=device, dtype=dtype)
#         )
#         assert res.keys() == expected.keys()
#         assert_allclose(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
#         assert_allclose(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
#         assert_allclose(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)

#     def test_same_on_batch(self, device, dtype):
#         torch.manual_seed(42)
#         angle = torch.tensor([30, 90])
#         direction = torch.tensor([-1, 1])
#         res = random_motion_blur_generator(
#             batch_size=2, kernel_size=3, angle=angle.to(device=device, dtype=dtype),
#             direction=direction.to(device=device, dtype=dtype), same_on_batch=True)
#         expected = dict(
#             ksize_factor=torch.tensor([3., 3.], device=device, dtype=dtype),
#             angle_factor=torch.tensor([33.4893, 33.4893], device=device, dtype=dtype),
#             direction_factor=torch.tensor([-0.8742, -0.8742], device=device, dtype=dtype)
#         )
#         assert res.keys() == expected.keys()
#         assert_allclose(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
#         assert_allclose(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
#         assert_allclose(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)
