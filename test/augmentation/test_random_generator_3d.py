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

    @pytest.mark.parametrize('batch_size', [0, 1, 8])
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
        pytest.param(-100, 100, 100, torch.tensor(0.5)),
        pytest.param(100, -100, 100, torch.tensor(0.5)),
        pytest.param(100, 100, -100, torch.tensor(-0.5)),
        pytest.param(100, 100, 100, torch.tensor(1.5)),
        pytest.param(100, 100, 100, torch.tensor([0., 0.5])),
    ])
    def test_invalid_param_combinations(self, depth, height, width, distortion_scale, device, dtype):
        with pytest.raises(Exception):
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
        assert_allclose(res['start_points'], expected['start_points'], atol=1e-4, rtol=1e-4)
        assert_allclose(res['end_points'], expected['end_points'], atol=1e-4, rtol=1e-4)


class TestRandomAffineGen3D(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [0, 1, 8])
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
        pytest.param(-100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, None),
        pytest.param(100, -100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, None),
        pytest.param(100, 100, -100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, None),
        pytest.param(100, 100, 100, torch.tensor([0, 9]), None, None, None),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     torch.tensor([0.1, 0.2]), None, None),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     torch.tensor([0.1, 0.2]), None, None),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     torch.tensor([0.1]), None, None),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     None, torch.tensor([[0.2, 0.2, 0.2]]), None),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     None, torch.tensor([0.2]), None),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     None, None, torch.tensor([[20, 20, 30]])),
        pytest.param(100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]),
                     None, None, torch.tensor([20])),
    ])
    def test_invalid_param_combinations(
        self, depth, height, width, degrees, translate, scale, shear, device, dtype
    ):
        with pytest.raises(Exception):
            random_affine_generator3d(
                batch_size=8, depth=depth, height=height, width=width,
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

    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('degrees', [torch.tensor([[0, 30], [0, 30], [0, 30]])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, degrees, same_on_batch, device, dtype):
        random_rotation_generator3d(
            batch_size=batch_size, degrees=degrees.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    @pytest.mark.parametrize('degrees', [
        pytest.param(torch.tensor(10)),
        pytest.param(torch.tensor([10])),
        pytest.param(torch.tensor([[0, 30]])),
        pytest.param(torch.tensor([[0, 30], [0, 30]])),
    ])
    def test_invalid_param_combinations(self, degrees, device, dtype):
        with pytest.raises(Exception):
            random_rotation_generator3d(
                batch_size=8, degrees=degrees.to(device=device, dtype=dtype))

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


class TestRandomCropGen3D(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [0, 2])
    @pytest.mark.parametrize('input_size', [(200, 200, 200)])
    @pytest.mark.parametrize('size', [(100, 100, 100), torch.tensor([50, 60, 70])])
    @pytest.mark.parametrize('resize_to', [None, (100, 100, 100)])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_size, input_size, size, resize_to, same_on_batch, device, dtype
    ):
        if isinstance(size, torch.Tensor):
            size = size.repeat(batch_size, 1).to(device=device, dtype=dtype)
        random_crop_generator3d(
            batch_size=batch_size, input_size=input_size,
            size=size.to(device=device, dtype=dtype) if isinstance(size, torch.Tensor) else size,
            resize_to=resize_to,
            same_on_batch=same_on_batch)

    @pytest.mark.parametrize('input_size,size,resize_to', [
        pytest.param((-300, 300, 300), (200, 200, 200), (100, 100, 100)),
        pytest.param((100, 100, 100), (200, 200, 200), (100, 100, 100)),
        pytest.param((200, 200, 200), torch.tensor([50, 50, 50]), (100, 100, 100)),
        pytest.param((100, 100, 100), torch.tensor([[50, 60, 70], [50, 60, 70]]),
                     (100, 100)),
    ])
    def test_invalid_param_combinations(self, input_size, size, resize_to, device, dtype):
        with pytest.raises(Exception):
            random_crop_generator3d(
                batch_size=2, input_size=input_size,
                size=size.to(device=device, dtype=dtype) if isinstance(size, torch.Tensor) else size,
                resize_to=resize_to)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        res = random_crop_generator3d(
            batch_size=2, input_size=(200, 200, 200),
            size=torch.tensor([[50, 60, 70], [50, 60, 70]], device=device, dtype=dtype),
            resize_to=(100, 100, 100))
        expected = dict(
            src=torch.tensor([[[7., 17., 79.],
                               [76., 17., 79.],
                               [76., 76., 79.],
                               [7., 76., 79.],
                               [7., 17., 128.],
                               [76., 17., 128.],
                               [76., 76., 128.],
                               [7., 76., 128.]],
                              [[8., 7., 71.],
                               [77., 7., 71.],
                               [77., 66., 71.],
                               [8., 66., 71.],
                               [8., 7., 120.],
                               [77., 7., 120.],
                               [77., 66., 120.],
                               [8., 66., 120.]]], device=device, dtype=torch.long),
            dst=torch.tensor([[[0., 0., 0.],
                               [99., 0., 0.],
                               [99., 99., 0.],
                               [0., 99., 0.],
                               [0., 0., 99.],
                               [99., 0., 99.],
                               [99., 99., 99.],
                               [0., 99., 99.]],
                              [[0., 0., 0.],
                               [99., 0., 0.],
                               [99., 99., 0.],
                               [0., 99., 0.],
                               [0., 0., 99.],
                               [99., 0., 99.],
                               [99., 99., 99.],
                               [0., 99., 99.]]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['src'], expected['src'], atol=1e-4, rtol=1e-4)
        assert_allclose(res['dst'], expected['dst'], atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        res = random_crop_generator3d(
            batch_size=2, input_size=(200, 200, 200),
            size=torch.tensor([[50, 60, 70], [50, 60, 70]], device=device, dtype=dtype),
            resize_to=(100, 100, 100), same_on_batch=True)
        expected = dict(
            src=torch.tensor([[[7., 8., 18.],
                               [76., 8., 18.],
                               [76., 67., 18.],
                               [7., 67., 18.],
                               [7., 8., 67.],
                               [76., 8., 67.],
                               [76., 67., 67.],
                               [7., 67., 67.]],
                              [[7., 8., 18.],
                               [76., 8., 18.],
                               [76., 67., 18.],
                               [7., 67., 18.],
                               [7., 8., 67.],
                               [76., 8., 67.],
                               [76., 67., 67.],
                               [7., 67., 67.]]], device=device, dtype=torch.long),
            dst=torch.tensor([[[0., 0., 0.],
                               [99., 0., 0.],
                               [99., 99., 0.],
                               [0., 99., 0.],
                               [0., 0., 99.],
                               [99., 0., 99.],
                               [99., 99., 99.],
                               [0., 99., 99.]],
                              [[0., 0., 0.],
                               [99., 0., 0.],
                               [99., 99., 0.],
                               [0., 99., 0.],
                               [0., 0., 99.],
                               [99., 0., 99.],
                               [99., 99., 99.],
                               [0., 99., 99.]]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['src'], expected['src'], atol=1e-4, rtol=1e-4)
        assert_allclose(res['dst'], expected['dst'], atol=1e-4, rtol=1e-4)


class TestCenterCropGen3D(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [0, 2])
    @pytest.mark.parametrize('depth,height,width', [(200, 200, 200)])
    @pytest.mark.parametrize('size', [(100, 100, 100)])
    def test_valid_param_combinations(
        self, batch_size, depth, height, width, size, device, dtype
    ):
        center_crop_generator3d(batch_size=batch_size, depth=depth, height=height, width=width, size=size)

    @pytest.mark.parametrize('depth,height,width,size', [
        pytest.param(200, 200, -200, (100, 100, 100)),
        pytest.param(200, -200, 200, (100, 100)),
        pytest.param(200, 100, 100, (300, 120, 100)),
        pytest.param(200, 150, 100, (120, 180, 100)),
        pytest.param(200, 100, 150, (120, 80, 200)),
    ])
    def test_invalid_param_combinations(self, depth, height, width, size, device, dtype):
        with pytest.raises(Exception):
            center_crop_generator3d(
                batch_size=2, depth=depth, height=height, width=width, size=size)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        res = center_crop_generator3d(batch_size=2, depth=200, height=200, width=200, size=(120, 150, 100))
        expected = dict(
            src=torch.tensor([
                [[50, 25, 40],
                 [149, 25, 40],
                 [149, 174, 40],
                 [50, 174, 40],
                 [50, 25, 159],
                 [149, 25, 159],
                 [149, 174, 159],
                 [50, 174, 159]]], device=device, dtype=torch.long),
            dst=torch.tensor([
                [[0, 0, 0],
                 [99, 0, 0],
                 [99, 149, 0],
                 [0, 149, 0],
                 [0, 0, 119],
                 [99, 0, 119],
                 [99, 149, 119],
                 [0, 149, 119]]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['src'], expected['src'], atol=1e-4, rtol=1e-4)
        assert_allclose(res['dst'], expected['dst'], atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        pass


class TestRandomMotionBlur3D(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('kernel_size', [3, (3, 5)])
    @pytest.mark.parametrize('angle', [torch.tensor([(10, 30), (30, 60), (60, 90)])])
    @pytest.mark.parametrize('direction', [torch.tensor([-1, -1]), torch.tensor([-1, 1]), torch.tensor([1, 1])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, kernel_size, angle, direction, same_on_batch, device, dtype):
        random_motion_blur_generator3d(
            batch_size=batch_size, kernel_size=kernel_size,
            angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype),
            same_on_batch=same_on_batch)

    @pytest.mark.parametrize('kernel_size,angle,direction', [
        pytest.param(4, torch.tensor([(10, 30), (30, 60), (60, 90)]), torch.tensor([-1, 1])),
        pytest.param(1, torch.tensor([(10, 30), (30, 60), (60, 90)]), torch.tensor([-1, 1])),
        pytest.param((3, 4, 5), torch.tensor([(10, 30), (30, 60), (60, 90)]),
                     torch.tensor([-1, 1])),
        pytest.param(3, torch.tensor([(10, 30), (30, 60), (60, 90)]), torch.tensor([-2, 1])),
        pytest.param(3, torch.tensor([(10, 30), (30, 60), (60, 90)]), torch.tensor([-1, 2])),
    ])
    def test_invalid_param_combinations(self, kernel_size, angle, direction, device, dtype):
        with pytest.raises(Exception):
            random_motion_blur_generator3d(
                batch_size=8, kernel_size=kernel_size, angle=angle.to(device=device, dtype=dtype),
                direction=direction.to(device=device, dtype=dtype))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        angle = torch.tensor([(10, 30), (30, 60), (60, 90)])
        direction = torch.tensor([-1, 1])
        res = random_motion_blur_generator3d(
            batch_size=2, kernel_size=3, angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype), same_on_batch=False)
        expected = dict(
            ksize_factor=torch.tensor([3., 3.], device=device, dtype=torch.int32),
            angle_factor=torch.tensor([
                [11.1631, 33.7076, 75.7852],
                [11.2582, 31.5774, 74.3035]], device=device, dtype=dtype),
            direction_factor=torch.tensor([0.9105, 0.8575], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        angle = torch.tensor([(10, 30), (30, 60), (60, 90)])
        direction = torch.tensor([-1, 1])
        res = random_motion_blur_generator3d(
            batch_size=2, kernel_size=3, angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            ksize_factor=torch.tensor([3., 3.], device=device, dtype=torch.int32),
            angle_factor=torch.tensor([
                [11.1631, 31.8873, 63.7076],
                [11.1631, 31.8873, 63.7076]], device=device, dtype=dtype),
            direction_factor=torch.tensor([-0.8948, -0.8948], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)
