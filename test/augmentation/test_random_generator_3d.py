import pytest
import torch

from kornia.augmentation.random_generator import (
    AffineGenerator3D,
    CropGenerator3D,
    MotionBlurGenerator3D,
    PerspectiveGenerator3D,
    RotationGenerator3D,
    center_crop_generator3d,
)
from kornia.testing import assert_close


class RandomGeneratorBaseTests:
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
    @pytest.mark.parametrize('distortion_scale', [torch.tensor(0.0), torch.tensor(0.5), torch.tensor(1.0)])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, depth, height, width, distortion_scale, batch_size, same_on_batch, device, dtype
    ):
        param_gen = PerspectiveGenerator3D(distortion_scale=distortion_scale.to(device=device, dtype=dtype))
        param_gen(batch_shape=torch.Size((batch_size, depth, height, width)), same_on_batch=same_on_batch)

    @pytest.mark.parametrize(
        'depth,height,width,distortion_scale',
        [
            # Should be failed if distortion_scale > 1. or distortion_scale < 0.
            (100, 100, -100, torch.tensor(-0.5)),
            (100, 100, 100, torch.tensor(1.5)),
            (100, 100, 100, torch.tensor([0.0, 0.5])),
        ],
    )
    def test_invalid_param_combinations(self, depth, height, width, distortion_scale, device, dtype):
        with pytest.raises(Exception):
            param_gen = PerspectiveGenerator3D(distortion_scale=distortion_scale.to(device=device, dtype=dtype))
            param_gen(batch_shape=torch.Size((2, depth, height, width)))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2

        param_gen = PerspectiveGenerator3D(distortion_scale=torch.tensor(0.5, device=device, dtype=dtype))
        res = param_gen(batch_shape=torch.Size((batch_size, 200, 200, 200)))

        expected = dict(
            start_points=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [199.0, 0.0, 0.0],
                        [199.0, 199.0, 0.0],
                        [0.0, 199.0, 0.0],
                        [0.0, 0.0, 199.0],
                        [199.0, 0.0, 199.0],
                        [199.0, 199.0, 199.0],
                        [0.0, 199.0, 199.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [199.0, 0.0, 0.0],
                        [199.0, 199.0, 0.0],
                        [0.0, 199.0, 0.0],
                        [0.0, 0.0, 199.0],
                        [199.0, 0.0, 199.0],
                        [199.0, 199.0, 199.0],
                        [0.0, 199.0, 199.0],
                    ],
                ],
                device=device,
                dtype=dtype,
            ),
            end_points=torch.tensor(
                [
                    [
                        [44.1135, 45.7502, 19.1432],
                        [151.0347, 19.5224, 30.0448],
                        [186.1714, 159.3179, 47.0386],
                        [6.6593, 152.2701, 29.6790],
                        [43.4702, 28.3858, 161.9453],
                        [177.5298, 44.2721, 170.3048],
                        [185.6710, 167.6275, 185.5184],
                        [22.0682, 184.1540, 157.4157],
                    ],
                    [
                        [5.2657, 13.4747, 17.9406],
                        [189.0318, 27.3596, 0.3080],
                        [151.4223, 195.2367, 44.3007],
                        [29.1605, 182.1176, 40.4487],
                        [28.8963, 45.1991, 171.2670],
                        [181.8843, 31.7171, 180.7795],
                        [163.4786, 151.6794, 159.5485],
                        [14.0707, 159.5684, 169.5268],
                    ],
                ],
                device=device,
                dtype=dtype,
            ),
        )
        assert res.keys() == expected.keys()
        assert_close(res['start_points'], expected['start_points'], atol=1e-4, rtol=1e-4)
        assert_close(res['end_points'], expected['end_points'], atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2

        param_gen = PerspectiveGenerator3D(distortion_scale=torch.tensor(0.5, device=device, dtype=dtype))
        res = param_gen(batch_shape=torch.Size((batch_size, 200, 200, 200)), same_on_batch=True)

        expected = dict(
            start_points=torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [199.0, 0.0, 0.0],
                        [199.0, 199.0, 0.0],
                        [0.0, 199.0, 0.0],
                        [0.0, 0.0, 199.0],
                        [199.0, 0.0, 199.0],
                        [199.0, 199.0, 199.0],
                        [0.0, 199.0, 199.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [199.0, 0.0, 0.0],
                        [199.0, 199.0, 0.0],
                        [0.0, 199.0, 0.0],
                        [0.0, 0.0, 199.0],
                        [199.0, 0.0, 199.0],
                        [199.0, 199.0, 199.0],
                        [0.0, 199.0, 199.0],
                    ],
                ],
                device=device,
                dtype=dtype,
            ),
            end_points=torch.tensor(
                [
                    [
                        [44.1135, 45.7502, 19.1432],
                        [151.0347, 19.5224, 30.0448],
                        [186.1714, 159.3179, 47.0386],
                        [6.6593, 152.2701, 29.6790],
                        [43.4702, 28.3858, 161.9453],
                        [177.5298, 44.2721, 170.3048],
                        [185.6710, 167.6275, 185.5184],
                        [22.0682, 184.1540, 157.4157],
                    ],
                    [
                        [44.1135, 45.7502, 19.1432],
                        [151.0347, 19.5224, 30.0448],
                        [186.1714, 159.3179, 47.0386],
                        [6.6593, 152.2701, 29.6790],
                        [43.4702, 28.3858, 161.9453],
                        [177.5298, 44.2721, 170.3048],
                        [185.6710, 167.6275, 185.5184],
                        [22.0682, 184.1540, 157.4157],
                    ],
                ],
                device=device,
                dtype=dtype,
            ),
        )
        assert res.keys() == expected.keys()
        assert_close(res['start_points'], expected['start_points'], atol=1e-4, rtol=1e-4)
        assert_close(res['end_points'], expected['end_points'], atol=1e-4, rtol=1e-4)


class TestRandomAffineGen3D(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_shape', [(0, 200, 300, 400), (1, 200, 300, 400), (8, 200, 300, 400)])
    @pytest.mark.parametrize('degrees', [torch.tensor([(0.0, 30.0), (0.0, 30.0), (0.0, 30.0)])])
    @pytest.mark.parametrize('translate', [None, torch.tensor([0.1, 0.1, 0.1])])
    @pytest.mark.parametrize(
        'scale', [None, torch.tensor([[0.7, 1.2], [0.7, 1.2], [0.7, 1.2]]), torch.tensor([0.7, 1.2])]
    )
    @pytest.mark.parametrize(
        'shear', [None, torch.tensor([[0.0, 20.0], [0.0, 20.0], [0.0, 20.0], [0.0, 20.0], [0.0, 20.0], [0.0, 20.0]])]
    )
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_shape, degrees, translate, scale, shear, same_on_batch, device, dtype
    ):
        if isinstance(degrees, torch.Tensor):
            degrees.to(dtype=dtype, device=device)
        if isinstance(translate, torch.Tensor):
            translate.to(dtype=dtype, device=device)
        if isinstance(scale, torch.Tensor):
            scale.to(dtype=dtype, device=device)
        if isinstance(shear, torch.Tensor):
            shear.to(dtype=dtype, device=device)

        param_gen = AffineGenerator3D(degrees=degrees, translate=translate, scale=scale, shears=shear)
        param_gen(batch_shape=torch.Size(batch_shape), same_on_batch=same_on_batch)

    @pytest.mark.parametrize(
        'depth,height,width,degrees,translate,scale,shear',
        [
            (-100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, None),
            (100, -100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, None),
            (100, 100, -100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, None),
            # (100, 100, 100, torch.tensor([0, 9]), None, None, None),
            (100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), torch.tensor([0.1, 0.2]), None, None),
            (100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), torch.tensor([0.1, 0.2]), None, None),
            (100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), torch.tensor([0.1]), None, None),
            (100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, torch.tensor([[0.2, 0.2, 0.2]]), None),
            (100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, torch.tensor([0.2]), None),
            (100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, torch.tensor([[20, 20, 30]])),
            # (100, 100, 100, torch.tensor([[0, 9], [0, 9], [0, 9]]), None, None, torch.tensor([20])),
        ],
    )
    def test_invalid_param_combinations(self, depth, height, width, degrees, translate, scale, shear, device, dtype):
        if isinstance(degrees, torch.Tensor):
            degrees.to(dtype=dtype, device=device)
        if isinstance(translate, torch.Tensor):
            translate.to(dtype=dtype, device=device)
        if isinstance(scale, torch.Tensor):
            scale.to(dtype=dtype, device=device)
        if isinstance(shear, torch.Tensor):
            shear.to(dtype=dtype, device=device)

        with pytest.raises(Exception):
            param_gen = AffineGenerator3D(degrees=degrees, translate=translate, scale=scale, shears=shear)
            param_gen(batch_shape=torch.Size((2, depth, height, width)))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([[10, 20], [10, 20], [10, 20]], device=device, dtype=dtype)
        translate = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=dtype)
        scale = torch.tensor([[0.7, 1.2], [0.7, 1.2], [0.7, 1.2]], device=device, dtype=dtype)
        shear = torch.tensor([[0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [0, 20]], device=device, dtype=dtype)

        param_gen = AffineGenerator3D(degrees=degrees, translate=translate, scale=scale, shears=shear)
        res = param_gen(batch_shape=torch.Size((2, 200, 200, 200)))

        expected = dict(
            translations=torch.tensor(
                [[14.7762, 9.6438, 15.4177], [2.7086, -2.8238, 2.9562]], device=device, dtype=dtype
            ),
            center=torch.tensor([[99.5000, 99.5000, 99.5000], [99.5000, 99.5000, 99.5000]], device=device, dtype=dtype),
            scale=torch.tensor([[0.8283, 1.1704, 1.1673], [1.0968, 0.7666, 0.9968]], device=device, dtype=dtype),
            angles=torch.tensor([[18.8227, 13.8286, 13.9045], [19.1500, 19.5931, 16.0090]], device=device, dtype=dtype),
            sxy=torch.tensor([5.3316, 12.5490], device=device, dtype=dtype),
            sxz=torch.tensor([5.3926, 8.8273], device=device, dtype=dtype),
            syx=torch.tensor([5.9384, 16.6337], device=device, dtype=dtype),
            syz=torch.tensor([2.1063, 5.3899], device=device, dtype=dtype),
            szx=torch.tensor([7.1763, 3.9873], device=device, dtype=dtype),
            szy=torch.tensor([10.9438, 0.1232], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['translations'], expected['translations'], rtol=1e-4, atol=1e-4)
        assert_close(res['center'], expected['center'], rtol=1e-4, atol=1e-4)
        assert_close(res['scale'], expected['scale'], rtol=1e-4, atol=1e-4)
        assert_close(res['angles'], expected['angles'], rtol=1e-4, atol=1e-4)
        assert_close(res['sxy'], expected['sxy'], rtol=1e-4, atol=1e-4)
        assert_close(res['sxz'], expected['sxz'], rtol=1e-4, atol=1e-4)
        assert_close(res['syx'], expected['syx'], rtol=1e-4, atol=1e-4)
        assert_close(res['syz'], expected['syz'], rtol=1e-4, atol=1e-4)
        assert_close(res['szx'], expected['szx'], rtol=1e-4, atol=1e-4)
        assert_close(res['szy'], expected['szy'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([[10, 20], [10, 20], [10, 20]], device=device, dtype=dtype)
        translate = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=dtype)
        scale = torch.tensor([[0.7, 1.2], [0.7, 1.2], [0.7, 1.2]], device=device, dtype=dtype)
        shear = torch.tensor([[0, 20], [0, 20], [0, 20], [0, 20], [0, 20], [0, 20]], device=device, dtype=dtype)

        param_gen = AffineGenerator3D(degrees=degrees, translate=translate, scale=scale, shears=shear)
        res = param_gen(batch_shape=torch.Size((2, 200, 200, 200)), same_on_batch=True)

        expected = dict(
            translations=torch.tensor(
                [[-9.7371, 11.7457, 17.6309], [-9.7371, 11.7457, 17.6309]], device=device, dtype=dtype
            ),
            center=torch.tensor([[99.5000, 99.5000, 99.5000], [99.5000, 99.5000, 99.5000]], device=device, dtype=dtype),
            scale=torch.tensor([[1.1797, 0.8952, 1.0004], [1.1797, 0.8952, 1.0004]], device=device, dtype=dtype),
            angles=torch.tensor([[18.8227, 19.1500, 13.8286], [18.8227, 19.1500, 13.8286]], device=device, dtype=dtype),
            sxy=torch.tensor([2.6637, 2.6637], device=device, dtype=dtype),
            sxz=torch.tensor([18.6920, 18.6920], device=device, dtype=dtype),
            syx=torch.tensor([11.8716, 11.8716], device=device, dtype=dtype),
            syz=torch.tensor([17.3881, 17.3881], device=device, dtype=dtype),
            szx=torch.tensor([11.3543, 11.3543], device=device, dtype=dtype),
            szy=torch.tensor([14.8219, 14.8219], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['translations'], expected['translations'], rtol=1e-4, atol=1e-4)
        assert_close(res['center'], expected['center'], rtol=1e-4, atol=1e-4)
        assert_close(res['scale'], expected['scale'], rtol=1e-4, atol=1e-4)
        assert_close(res['angles'], expected['angles'], rtol=1e-4, atol=1e-4)
        assert_close(res['sxy'], expected['sxy'], rtol=1e-4, atol=1e-4)
        assert_close(res['sxz'], expected['sxz'], rtol=1e-4, atol=1e-4)
        assert_close(res['syx'], expected['syx'], rtol=1e-4, atol=1e-4)
        assert_close(res['syz'], expected['syz'], rtol=1e-4, atol=1e-4)
        assert_close(res['szx'], expected['szx'], rtol=1e-4, atol=1e-4)
        assert_close(res['szy'], expected['szy'], rtol=1e-4, atol=1e-4)


class TestRandomRotationGen3D(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, same_on_batch, device, dtype):
        degrees = torch.tensor([[0.0, 30.0], [0.0, 30.0], [0.0, 30.0]], device=device, dtype=dtype)
        param_gen = RotationGenerator3D(degrees=degrees.to(device=device, dtype=dtype))
        param_gen(torch.Size((batch_size,)), same_on_batch=same_on_batch)

    @pytest.mark.parametrize(
        'degrees',
        [(torch.tensor(-10)), (torch.tensor([-10])), (torch.tensor([[0, 30]])), (torch.tensor([[0, 30], [0, 30]]))],
    )
    def test_invalid_param_combinations(self, degrees, device, dtype):
        with pytest.raises(Exception):
            param_gen = RotationGenerator3D(degrees=degrees.to(device=device, dtype=dtype))
            param_gen(torch.Size((2,)))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([[0.0, 30.0], [0.0, 30.0], [0.0, 30.0]], device=device, dtype=dtype)
        param_gen = RotationGenerator3D(degrees=degrees)
        res = param_gen(torch.Size((2,)), same_on_batch=False)

        expected = dict(
            yaw=torch.tensor([26.4681, 27.4501], device=device, dtype=dtype),
            pitch=torch.tensor([11.4859, 28.7792], device=device, dtype=dtype),
            roll=torch.tensor([11.7134, 18.0269], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['yaw'], expected['yaw'], atol=1e-4, rtol=1e-4)
        assert_close(res['pitch'], expected['pitch'], atol=1e-4, rtol=1e-4)
        assert_close(res['roll'], expected['roll'], atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([[0.0, 30.0], [0.0, 30.0], [0.0, 30.0]], device=device, dtype=dtype)
        param_gen = RotationGenerator3D(degrees=degrees)
        res = param_gen(torch.Size((2,)), same_on_batch=True)

        expected = dict(
            yaw=torch.tensor([26.4681, 26.4681], device=device, dtype=dtype),
            pitch=torch.tensor([27.4501, 27.4501], device=device, dtype=dtype),
            roll=torch.tensor([11.4859, 11.4859], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['yaw'], expected['yaw'], atol=1e-4, rtol=1e-4)
        assert_close(res['pitch'], expected['pitch'], atol=1e-4, rtol=1e-4)
        assert_close(res['roll'], expected['roll'], atol=1e-4, rtol=1e-4)


class TestRandomCropGen3D(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 2])
    @pytest.mark.parametrize('input_size', [(200, 200, 200)])
    @pytest.mark.parametrize('size', [(100, 100, 100), torch.tensor([50, 60, 70])])
    @pytest.mark.parametrize('resize_to', [None, (100, 100, 100)])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, input_size, size, resize_to, same_on_batch, device, dtype):
        if isinstance(size, torch.Tensor):
            size = size.repeat(batch_size, 1).to(device=device, dtype=dtype)

        param_gen = CropGenerator3D(size=size, resize_to=resize_to)
        param_gen(batch_shape=torch.Size((batch_size, 1, *input_size)), same_on_batch=same_on_batch)

    @pytest.mark.parametrize(
        'input_size,size,resize_to',
        [
            ((-300, 300, 300), (200, 200, 200), (100, 100, 100)),
            ((100, 100, 100), (200, 200, 200), (100, 100, 100)),
            ((200, 200, 200), torch.tensor([50, 50, 50]), (100, 100, 100)),
            ((100, 100, 100), torch.tensor([[50, 60, 70], [50, 60, 70]]), (100, 100)),
        ],
    )
    def test_invalid_param_combinations(self, input_size, size, resize_to, device, dtype):
        with pytest.raises(Exception):
            param_gen = CropGenerator3D(
                size=size.to(device=device, dtype=dtype) if isinstance(size, torch.Tensor) else size,
                resize_to=resize_to,
            )
            param_gen(batch_shape=torch.Size((2, 1, *input_size)))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        param_gen = CropGenerator3D(
            size=torch.tensor([[50, 60, 70], [50, 60, 70]], device=device, dtype=dtype), resize_to=(100, 100, 100)
        )
        res = param_gen(batch_shape=torch.Size((2, 1, 200, 200, 200)))

        expected = dict(
            src=torch.tensor(
                [
                    [
                        [115, 53, 58],
                        [184, 53, 58],
                        [184, 112, 58],
                        [115, 112, 58],
                        [115, 53, 107],
                        [184, 53, 107],
                        [184, 112, 107],
                        [115, 112, 107],
                    ],
                    [
                        [119, 135, 90],
                        [188, 135, 90],
                        [188, 194, 90],
                        [119, 194, 90],
                        [119, 135, 139],
                        [188, 135, 139],
                        [188, 194, 139],
                        [119, 194, 139],
                    ],
                ],
                device=device,
                dtype=dtype,
            ),
            dst=torch.tensor(
                [
                    [
                        [0, 0, 0],
                        [99, 0, 0],
                        [99, 99, 0],
                        [0, 99, 0],
                        [0, 0, 99],
                        [99, 0, 99],
                        [99, 99, 99],
                        [0, 99, 99],
                    ],
                    [
                        [0, 0, 0],
                        [99, 0, 0],
                        [99, 99, 0],
                        [0, 99, 0],
                        [0, 0, 99],
                        [99, 0, 99],
                        [99, 99, 99],
                        [0, 99, 99],
                    ],
                ],
                device=device,
                dtype=dtype,
            ),
        )
        assert res.keys() == expected.keys()
        assert_close(res['src'], expected['src'], atol=1e-4, rtol=1e-4)
        assert_close(res['dst'], expected['dst'], atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)

        param_gen = CropGenerator3D(
            size=torch.tensor([[50, 60, 70], [50, 60, 70]], device=device, dtype=dtype), resize_to=(100, 100, 100)
        )
        res = param_gen(batch_shape=torch.Size((2, 1, 200, 200, 200)), same_on_batch=True)

        expected = dict(
            src=torch.tensor(
                [
                    [
                        [115, 129, 57],
                        [184, 129, 57],
                        [184, 188, 57],
                        [115, 188, 57],
                        [115, 129, 106],
                        [184, 129, 106],
                        [184, 188, 106],
                        [115, 188, 106],
                    ],
                    [
                        [115, 129, 57],
                        [184, 129, 57],
                        [184, 188, 57],
                        [115, 188, 57],
                        [115, 129, 106],
                        [184, 129, 106],
                        [184, 188, 106],
                        [115, 188, 106],
                    ],
                ],
                device=device,
                dtype=dtype,
            ),
            dst=torch.tensor(
                [
                    [
                        [0, 0, 0],
                        [99, 0, 0],
                        [99, 99, 0],
                        [0, 99, 0],
                        [0, 0, 99],
                        [99, 0, 99],
                        [99, 99, 99],
                        [0, 99, 99],
                    ],
                    [
                        [0, 0, 0],
                        [99, 0, 0],
                        [99, 99, 0],
                        [0, 99, 0],
                        [0, 0, 99],
                        [99, 0, 99],
                        [99, 99, 99],
                        [0, 99, 99],
                    ],
                ],
                device=device,
                dtype=dtype,
            ),
        )
        assert res.keys() == expected.keys()
        assert_close(res['src'], expected['src'], atol=1e-4, rtol=1e-4)
        assert_close(res['dst'], expected['dst'], atol=1e-4, rtol=1e-4)


class TestCenterCropGen3D(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 2])
    @pytest.mark.parametrize('depth,height,width', [(200, 200, 200)])
    @pytest.mark.parametrize('size', [(100, 100, 100)])
    def test_valid_param_combinations(self, batch_size, depth, height, width, size, device, dtype):
        center_crop_generator3d(batch_size=batch_size, depth=depth, height=height, width=width, size=size)

    @pytest.mark.parametrize(
        'depth,height,width,size',
        [
            (200, 200, -200, (100, 100, 100)),
            (200, -200, 200, (100, 100)),
            (200, 100, 100, (300, 120, 100)),
            (200, 150, 100, (120, 180, 100)),
            (200, 100, 150, (120, 80, 200)),
        ],
    )
    def test_invalid_param_combinations(self, depth, height, width, size, device, dtype):
        with pytest.raises(Exception):
            center_crop_generator3d(batch_size=2, depth=depth, height=height, width=width, size=size)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        res = center_crop_generator3d(batch_size=2, depth=200, height=200, width=200, size=(120, 150, 100))
        expected = dict(
            src=torch.tensor(
                [
                    [
                        [50, 25, 40],
                        [149, 25, 40],
                        [149, 174, 40],
                        [50, 174, 40],
                        [50, 25, 159],
                        [149, 25, 159],
                        [149, 174, 159],
                        [50, 174, 159],
                    ]
                ],
                device=device,
                dtype=torch.long,
            ).repeat(2, 1, 1),
            dst=torch.tensor(
                [
                    [
                        [0, 0, 0],
                        [99, 0, 0],
                        [99, 149, 0],
                        [0, 149, 0],
                        [0, 0, 119],
                        [99, 0, 119],
                        [99, 149, 119],
                        [0, 149, 119],
                    ]
                ],
                device=device,
                dtype=torch.long,
            ).repeat(2, 1, 1),
        )
        assert res.keys() == expected.keys()
        assert_close(res['src'].to(device=device), expected['src'], atol=1e-4, rtol=1e-4)
        assert_close(res['dst'].to(device=device), expected['dst'], atol=1e-4, rtol=1e-4)

    def test_same_on_batch(self, device, dtype):
        pass


class TestRandomMotionBlur3D(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('kernel_size', [3, (3, 5)])
    @pytest.mark.parametrize('angle', [torch.tensor([(10.0, 30.0), (30.0, 60.0), (60.0, 90.0)])])
    @pytest.mark.parametrize(
        'direction', [torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, 1.0]), torch.tensor([1.0, 1.0])]
    )
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, kernel_size, angle, direction, same_on_batch, device, dtype):
        param_gen = MotionBlurGenerator3D(
            kernel_size=kernel_size,
            angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype),
        )

        param_gen(batch_shape=torch.Size((batch_size,)), same_on_batch=same_on_batch)

    @pytest.mark.parametrize(
        'kernel_size,angle,direction',
        [
            (4, torch.tensor([(10, 30), (30, 60), (60, 90)]), torch.tensor([-1, 1])),
            (1, torch.tensor([(10, 30), (30, 60), (60, 90)]), torch.tensor([-1, 1])),
            ((3, 4, 5), torch.tensor([(10, 30), (30, 60), (60, 90)]), torch.tensor([-1, 1])),
            (3, torch.tensor([(10, 30), (30, 60), (60, 90)]), torch.tensor([-2, 1])),
            (3, torch.tensor([(10, 30), (30, 60), (60, 90)]), torch.tensor([-1, 2])),
        ],
    )
    def test_invalid_param_combinations(self, kernel_size, angle, direction, device, dtype):
        with pytest.raises(Exception):
            param_gen = MotionBlurGenerator3D(
                kernel_size=kernel_size,
                angle=angle.to(device=device, dtype=dtype),
                direction=direction.to(device=device, dtype=dtype),
            )

            param_gen(batch_shape=torch.Size((2,)))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        angle = torch.tensor([(10, 30), (30, 60), (60, 90)], device=device, dtype=dtype)
        direction = torch.tensor([-1, 1], device=device, dtype=dtype)
        param_gen = MotionBlurGenerator3D(kernel_size=3, angle=angle, direction=direction)

        res = param_gen(batch_shape=torch.Size((2,)), same_on_batch=False)

        expected = dict(
            ksize_factor=torch.tensor([3, 3], device=device, dtype=torch.int32),
            angle_factor=torch.tensor(
                [[27.6454, 41.4859, 71.7134], [28.3001, 58.7792, 78.0269]], device=device, dtype=dtype
            ),
            direction_factor=torch.tensor([-0.4869, 0.5873], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
        assert_close(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
        assert_close(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        angle = torch.tensor([(10, 30), (30, 60), (60, 90)], device=device, dtype=dtype)
        direction = torch.tensor([-1, 1], device=device, dtype=dtype)
        param_gen = MotionBlurGenerator3D(kernel_size=3, angle=angle, direction=direction)

        res = param_gen(batch_shape=torch.Size((2,)), same_on_batch=True)

        expected = dict(
            ksize_factor=torch.tensor([3, 3], device=device, dtype=torch.int32),
            angle_factor=torch.tensor(
                [[27.6454, 57.4501, 71.4859], [27.6454, 57.4501, 71.4859]], device=device, dtype=dtype
            ),
            direction_factor=torch.tensor([0.9186, 0.9186], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
        assert_close(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
        assert_close(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)
