
import pytest
import torch
from torch.testing import assert_allclose

from kornia.augmentation.random_generator import (
    random_prob_generator,
    random_color_jitter_generator,
    random_perspective_generator,
    random_affine_generator,
    random_rotation_generator,
    random_crop_generator,
    random_crop_size_generator,
    random_rectangles_params_generator,
    center_crop_generator,
    random_motion_blur_generator,
    random_solarize_generator,
    random_posterize_generator,
    random_sharpness_generator,
    random_mixup_generator,
    random_cutmix_generator,
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


class TestRandomProbGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('p', [0., 0.5, 1.])
    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, p, batch_size, same_on_batch, device, dtype):
        random_prob_generator(batch_size=batch_size, p=p, same_on_batch=same_on_batch)

    @pytest.mark.parametrize('p', [
        # Should be failed if p > 1. or p < 0.
        pytest.param(-1., marks=pytest.mark.xfail),
        pytest.param(2., marks=pytest.mark.xfail)
    ])
    def test_invalid_param_combinations(self, p, device, dtype):
        random_prob_generator(batch_size=8, p=p)

    @pytest.mark.parametrize('p,expected', [
        (0., [False] * 8),
        (0.5, [False, False, True, False, True, False, True, False]),
        (1., [True] * 8)
    ])
    def test_random_gen(self, p, expected, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = random_prob_generator(batch_size=batch_size, p=p)
        assert (res == torch.tensor(expected)).long().sum() == batch_size

    @pytest.mark.parametrize("seed,expected", [
        (42, [False] * 8),
        (0, [True] * 8),
    ])
    def test_same_on_batch(self, seed, expected, device, dtype):
        torch.manual_seed(seed)
        batch_size = 8
        res = random_prob_generator(batch_size=batch_size, p=.5, same_on_batch=True)
        assert (res == torch.tensor(expected)).long().sum() == batch_size


class TestColorJitterGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('brightness', [None, torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize('contrast', [None, torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize('saturation', [None, torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize('hue', [None, torch.tensor([-0.1, 0.1])])
    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, brightness, contrast, saturation, hue, batch_size, same_on_batch, device, dtype
    ):
        random_color_jitter_generator(
            batch_size,
            brightness.to(device=device, dtype=dtype) if brightness is not None else None,
            contrast.to(device=device, dtype=dtype) if contrast is not None else None,
            saturation.to(device=device, dtype=dtype) if saturation is not None else None,
            hue.to(device=device, dtype=dtype) if hue is not None else None,
            same_on_batch
        )

    @pytest.mark.parametrize('brightness,contrast,saturation,hue', [
        # Should be failed if value out of bounds or tensor.shape != [1, 2]
        pytest.param(torch.tensor([-1., 2.]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(torch.tensor([0., 3.]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(torch.tensor(0.), None, None, None, marks=pytest.mark.xfail),
        pytest.param(torch.tensor([0.]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(torch.tensor([0., 1., 2.]), None, None, None, marks=pytest.mark.xfail),

        pytest.param(None, torch.tensor([-1., 2.]), None, None, marks=pytest.mark.xfail),
        pytest.param(None, torch.tensor(0.), None, None, marks=pytest.mark.xfail),
        pytest.param(None, torch.tensor([0.]), None, None, marks=pytest.mark.xfail),
        pytest.param(None, torch.tensor([0., 1., 2.]), None, None, marks=pytest.mark.xfail),

        pytest.param(None, None, torch.tensor([-1., 2.]), None, marks=pytest.mark.xfail),
        pytest.param(None, None, torch.tensor(0.), None, marks=pytest.mark.xfail),
        pytest.param(None, None, torch.tensor([0.]), None, marks=pytest.mark.xfail),
        pytest.param(None, None, torch.tensor([0., 1., 2.]), None, marks=pytest.mark.xfail),

        pytest.param(None, None, None, torch.tensor([-1., 0.]), marks=pytest.mark.xfail),
        pytest.param(None, None, None, torch.tensor([0, 1.]), marks=pytest.mark.xfail),
        pytest.param(None, None, None, torch.tensor(0.), marks=pytest.mark.xfail),
        pytest.param(None, None, None, torch.tensor([0.]), marks=pytest.mark.xfail),
        pytest.param(None, None, None, torch.tensor([0., 1., 2.]), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(
        self, brightness, contrast, saturation, hue, device, dtype
    ):
        random_color_jitter_generator(
            8,
            brightness.to(device=device, dtype=dtype) if brightness is not None else None,
            contrast.to(device=device, dtype=dtype) if contrast is not None else None,
            saturation.to(device=device, dtype=dtype) if saturation is not None else None,
            hue.to(device=device, dtype=dtype) if hue is not None else None
        )

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = random_color_jitter_generator(
            batch_size, brightness=torch.tensor([0.8, 1.2], device=device, dtype=dtype),
            contrast=torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            saturation=torch.tensor([0.6, 1.4], device=device, dtype=dtype),
            hue=torch.tensor([-0.1, 0.1], device=device, dtype=dtype))

        expected_jitter_params = {
            'brightness_factor': torch.tensor(
                [0.8233, 0.8252, 0.8494, 0.8210, 1.0105, 0.9907, 1.1821, 1.1715], device=device, dtype=dtype),
            'contrast_factor': torch.tensor(
                [0.7501, 0.7796, 0.7942, 0.9252, 1.2055, 1.2223, 0.9265, 1.0675], device=device, dtype=dtype),
            'hue_factor': torch.tensor(
                [-0.0824, 0.0402, 0.0247, -0.0125, -0.0851, 0.0367, -0.0376, -0.0382], device=device, dtype=dtype),
            'saturation_factor': torch.tensor(
                [0.6251, 0.6323, 1.3455, 0.7217, 0.8120, 0.7043, 0.8015, 0.7867], device=device, dtype=dtype),
            'order': torch.tensor([2, 3, 1, 0], device=device, dtype=dtype)
        }

        assert set(list(jitter_params.keys())) == set([
            'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor', 'order']), \
            "Redundant keys found apart from \
                'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor', 'order'"

        assert_allclose(
            jitter_params['brightness_factor'], expected_jitter_params['brightness_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(
            jitter_params['contrast_factor'], expected_jitter_params['contrast_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(
            jitter_params['hue_factor'], expected_jitter_params['hue_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(
            jitter_params['saturation_factor'], expected_jitter_params['saturation_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(
            jitter_params['order'].to(dtype), expected_jitter_params['order'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = random_color_jitter_generator(
            batch_size, brightness=torch.tensor([0.8, 1.2], device=device, dtype=dtype),
            contrast=torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            saturation=torch.tensor([0.6, 1.4], device=device, dtype=dtype),
            hue=torch.tensor([-0.1, 0.1], device=device, dtype=dtype), same_on_batch=True)

        expected_res = {
            'brightness_factor': torch.tensor([0.8233] * batch_size, device=device, dtype=dtype),
            'contrast_factor': torch.tensor([0.7377] * batch_size, device=device, dtype=dtype),
            'hue_factor': torch.tensor([-0.0753] * batch_size, device=device, dtype=dtype),
            'saturation_factor': torch.tensor([0.6421] * batch_size, device=device, dtype=dtype),
            'order': torch.tensor([2, 3, 0, 1], device=device, dtype=dtype)
        }

        assert_allclose(
            jitter_params['brightness_factor'], expected_res['brightness_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(
            jitter_params['contrast_factor'], expected_res['contrast_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(
            jitter_params['hue_factor'], expected_res['hue_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(
            jitter_params['saturation_factor'], expected_res['saturation_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(
            jitter_params['order'].to(dtype), expected_res['order'], rtol=1e-4, atol=1e-4)


class TestRandomPerspectiveGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('height,width', [(200, 200)])
    @pytest.mark.parametrize('distortion_scale', [torch.tensor(0.), torch.tensor(0.5), torch.tensor(1.)])
    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, height, width, distortion_scale, batch_size, same_on_batch, device, dtype
    ):
        random_perspective_generator(
            batch_size=8, height=height, width=width,
            distortion_scale=distortion_scale.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    @pytest.mark.parametrize('height,width,distortion_scale', [
        # Should be failed if distortion_scale > 1. or distortion_scale < 0.
        pytest.param(-100, 100, torch.tensor(0.5), marks=pytest.mark.xfail),
        pytest.param(100, -100, torch.tensor(0.5), marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor(-0.5), marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor(1.5), marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([0., 0.5]), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, height, width, distortion_scale, device, dtype):
        random_perspective_generator(
            batch_size=8, height=height, width=width,
            distortion_scale=distortion_scale.to(device=device, dtype=dtype))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        res = random_perspective_generator(batch_size, 200, 200, torch.tensor(0.5, device=device, dtype=dtype))
        expected = dict(
            start_points=torch.tensor([
                [[0., 0.],
                 [199., 0.],
                 [199., 199.],
                 [0., 199.]],
                [[0., 0.],
                 [199., 0.],
                 [199., 199.],
                 [0., 199.]]], device=device, dtype=dtype),
            end_points=torch.tensor([
                [[2.9077, 3.1455],
                 [192.8207, 2.6290],
                 [172.6914, 175.1608],
                 [47.7618, 152.5624]],
                [[4.1772, 6.6320],
                 [191.1473, 18.7684],
                 [156.8740, 155.4749],
                 [18.8766, 168.3782]]], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['start_points'], expected['start_points'])
        assert_allclose(res['end_points'], expected['end_points'])

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        res = random_perspective_generator(
            batch_size, 200, 200, torch.tensor(0.5, device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            start_points=torch.tensor([
                [[0., 0.],
                 [199., 0.],
                 [199., 199.],
                 [0., 199.]]], device=device, dtype=dtype).repeat(2, 1, 1),
            end_points=torch.tensor([
                [[2.9077, 3.1455],
                 [192.8207, 2.6290],
                 [172.6914, 175.1608],
                 [47.7618, 152.5624]]], device=device, dtype=dtype).repeat(2, 1, 1),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['start_points'], expected['start_points'])
        assert_allclose(res['end_points'], expected['end_points'])


class TestRandomAffineGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('height', [200])
    @pytest.mark.parametrize('width', [300])
    @pytest.mark.parametrize('degrees', [torch.tensor([0, 30])])
    @pytest.mark.parametrize('translate', [None, torch.tensor([0.1, 0.1])])
    @pytest.mark.parametrize('scale', [None, torch.tensor([0.7, 1.2])])
    @pytest.mark.parametrize('shear', [None, torch.tensor([[0, 20], [0, 20]])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_size, height, width, degrees, translate, scale, shear, same_on_batch, device, dtype
    ):
        random_affine_generator(
            batch_size=batch_size, height=height, width=width, degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shear=shear.to(device=device, dtype=dtype) if shear is not None else None,
            same_on_batch=same_on_batch)

    @pytest.mark.parametrize('height,width,degrees,translate,scale,shear', [
        pytest.param(-100, 100, torch.tensor([10, 20]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(100, -100, torch.tensor([10, 20]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, 0.5, None, None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([10, 20, 30]), None, None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([10, 20]), torch.tensor([0.1]), None, None, marks=pytest.mark.xfail),
        pytest.param(10, 10, torch.tensor([1, 2]), torch.tensor([0.1, 0.2, 0.3]), None, None, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([10, 20]), None, torch.tensor([1]), None, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([10, 20]), None, torch.tensor([1, 2, 3]), None, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([10, 20]), None, None, torch.tensor([1]), marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([10, 20]), None, None, torch.tensor([1, 2]), marks=pytest.mark.xfail),
        pytest.param(10, 10, torch.tensor([1, 2]), None, None, torch.tensor([1, 2, 3]), marks=pytest.mark.xfail),
        pytest.param(10, 10, torch.tensor([1, 2]), None, None, torch.tensor([1, 2, 3, 4]), marks=pytest.mark.xfail),
        pytest.param(10, 10, torch.tensor([1, 2]), None, None, torch.tensor([1, 2, 3, 4, 5]), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(
        self, batch_size, height, width, degrees, translate, scale, shear, device, dtype
    ):
        batch_size = 8
        random_affine_generator(
            batch_size=batch_size, height=height, width=width, degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shear=shear.to(device=device, dtype=dtype) if shear is not None else None)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20])
        translate = torch.tensor([0.1, 0.1])
        scale = torch.tensor([0.7, 1.2])
        shear = torch.tensor([[10, 20], [10, 20]])
        res = random_affine_generator(
            batch_size=2, height=200, width=200, degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shear=shear.to(device=device, dtype=dtype) if shear is not None else None,
            same_on_batch=False)
        expected = dict(
            translations=torch.tensor([[1.0469, 18.2094], [-0.9286, 17.1501]], device=device, dtype=dtype),
            center=torch.tensor([[99.5000, 99.5000], [99.5000, 99.5000]], device=device, dtype=dtype),
            scale=torch.tensor([[0.7618, 0.7618], [0.7263, 0.7263]], device=device, dtype=dtype),
            angle=torch.tensor([10.5815, 10.6291], device=device, dtype=dtype),
            sx=torch.tensor([10.8354, 11.3264], device=device, dtype=dtype),
            sy=torch.tensor([11.5705, 13.7537], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['translations'], expected['translations'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['center'], expected['center'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['scale'], expected['scale'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['angle'], expected['angle'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['sx'], expected['sx'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['sy'], expected['sy'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20])
        translate = torch.tensor([0.1, 0.1])
        scale = torch.tensor([0.7, 1.2])
        shear = torch.tensor([[10, 20], [10, 20]])
        res = random_affine_generator(
            batch_size=2, height=200, width=200, degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shear=shear.to(device=device, dtype=dtype) if shear is not None else None,
            same_on_batch=True)
        expected = dict(
            translations=torch.tensor([[-15.0566, -17.8968], [-15.0566, -17.8968]], device=device, dtype=dtype),
            center=torch.tensor([[99.5000, 99.5000], [99.5000, 99.5000]], device=device, dtype=dtype),
            scale=torch.tensor([[0.7315, 0.7315], [0.7315, 0.7315]], device=device, dtype=dtype),
            angle=torch.tensor([10.5815, 10.5815], device=device, dtype=dtype),
            sx=torch.tensor([15.2617, 15.2617], device=device, dtype=dtype),
            sy=torch.tensor([14.7678, 14.7678], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['translations'], expected['translations'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['center'], expected['center'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['scale'], expected['scale'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['angle'], expected['angle'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['sx'], expected['sx'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['sy'], expected['sy'], rtol=1e-4, atol=1e-4)


class TestRandomRotationGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('degrees', [torch.tensor([0, 30])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, degrees, same_on_batch, device, dtype):
        random_rotation_generator(
            batch_size=batch_size, degrees=degrees.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    @pytest.mark.parametrize('degrees', [
        pytest.param(torch.tensor(10), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([10]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([10, 20, 30]), marks=pytest.mark.xfail)
    ])
    def test_invalid_param_combinations(self, degrees, device, dtype):
        batch_size = 8
        random_rotation_generator(
            batch_size=batch_size, degrees=degrees.to(device=device, dtype=dtype))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20])
        res = random_rotation_generator(
            batch_size=2, degrees=degrees.to(device=device, dtype=dtype), same_on_batch=False)
        expected = dict(
            degrees=torch.tensor([10.5815, 10.6291], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['degrees'], expected['degrees'])

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20])
        res = random_rotation_generator(
            batch_size=2, degrees=degrees.to(device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            degrees=torch.tensor([10.5815, 10.5815], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['degrees'], expected['degrees'])


class TestRandomCropGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [2])
    @pytest.mark.parametrize('input_size', [(200, 200)])
    @pytest.mark.parametrize('size', [(100, 100), torch.tensor([[50, 50], [60, 60]])])
    @pytest.mark.parametrize('resize_to', [None, (100, 100)])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_size, input_size, size, resize_to, same_on_batch, device, dtype
    ):
        random_crop_generator(
            batch_size=batch_size, input_size=input_size,
            size=size.to(device=device, dtype=dtype) if isinstance(size, torch.Tensor) else size,
            resize_to=resize_to,
            same_on_batch=same_on_batch)

    @pytest.mark.parametrize('input_size,size,resize_to', [
        pytest.param((-300, 300), (200, 200), (100, 100), marks=pytest.mark.xfail),
        pytest.param((100, 100), (200, 200), (100, 100), marks=pytest.mark.xfail),
        pytest.param((200, 200), torch.tensor([50, 50]), (100, 100), marks=pytest.mark.xfail),
        pytest.param((100, 100), torch.tensor([[200, 200], [200, 200]]), (100, 100), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, input_size, size, resize_to, device, dtype):
        batch_size = 2
        random_crop_generator(
            batch_size=batch_size, input_size=input_size,
            size=size.to(device=device, dtype=dtype) if isinstance(size, torch.Tensor) else size,
            resize_to=resize_to)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20])
        res = random_crop_generator(
            batch_size=2, input_size=(100, 100),
            size=torch.tensor([[50, 60], [70, 80]], device=device, dtype=dtype),
            resize_to=(200, 200))
        expected = dict(
            src=torch.tensor([
                [[2, 6],
                 [61, 6],
                 [61, 55],
                 [2, 55]],
                [[1, 1],
                 [80, 1],
                 [80, 70],
                 [1, 70]]], device=device, dtype=torch.long),
            dst=torch.tensor([
                [[0, 0],
                 [199, 0],
                 [199, 199],
                 [0, 199]],
                [[0, 0],
                 [199, 0],
                 [199, 199],
                 [0, 199]]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['src'], expected['src'])
        assert_allclose(res['dst'], expected['dst'])

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20])
        res = random_crop_generator(
            batch_size=2, input_size=(100, 100),
            size=torch.tensor([[50, 60], [70, 80]], device=device, dtype=dtype),
            resize_to=(200, 200), same_on_batch=True)
        expected = dict(
            src=torch.tensor([
                [[2, 3],
                 [61, 3],
                 [61, 52],
                 [2, 52]],
                [[2, 3],
                 [81, 3],
                 [81, 72],
                 [2, 72]]], device=device, dtype=torch.long),
            dst=torch.tensor([
                [[0, 0],
                 [199, 0],
                 [199, 199],
                 [0, 199]],
                [[0, 0],
                 [199, 0],
                 [199, 199],
                 [0, 199]]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['src'], expected['src'])
        assert_allclose(res['dst'], expected['dst'])


class TestRandomCropSizeGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('size', [(200, 200)])
    @pytest.mark.parametrize('scale', [torch.tensor([.7, 1.3])])
    @pytest.mark.parametrize('ratio', [torch.tensor([.9, 1.1])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_size, size, scale, ratio, same_on_batch, device, dtype
    ):
        random_crop_size_generator(
            batch_size=batch_size, size=size,
            scale=scale.to(device=device, dtype=dtype),
            ratio=ratio.to(device=device, dtype=dtype),
            same_on_batch=same_on_batch)

    @pytest.mark.parametrize('size,scale,ratio', [
        pytest.param((100), torch.tensor([.7, 1.3]), torch.tensor([.9, 1.1]), marks=pytest.mark.xfail),
        pytest.param((100, 100, 100), torch.tensor([.7, 1.3]), torch.tensor([.9, 1.1]), marks=pytest.mark.xfail),
        pytest.param((100, 100), torch.tensor([.7]), torch.tensor([.9, 1.1]), marks=pytest.mark.xfail),
        pytest.param((100, 100), torch.tensor([.7, 1.3, 1.5]), torch.tensor([.9, 1.1]), marks=pytest.mark.xfail),
        pytest.param((100, 100), torch.tensor([.7, 1.3]), torch.tensor([.9]), marks=pytest.mark.xfail),
        pytest.param((100, 100), torch.tensor([.7, 1.3]), torch.tensor([.9, 1.1, 1.3]), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, size, scale, ratio, device, dtype):
        batch_size = 2
        random_crop_size_generator(
            batch_size=batch_size, size=size,
            scale=scale.to(device=device, dtype=dtype),
            ratio=ratio.to(device=device, dtype=dtype),
            same_on_batch=same_on_batch)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20])
        res = random_crop_size_generator(
            batch_size=8, size=(100, 100),
            scale=torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            ratio=torch.tensor([0.9, 1.1], device=device, dtype=dtype),
            same_on_batch=False)
        expected = dict(
            size=torch.tensor([
                [84, 92],
                [87, 86],
                [89, 87],
                [98, 97],
                [89, 84],
                [90, 96],
                [89, 87],
                [92, 95]], device=device, dtype=torch.long)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['size'], expected['size'])

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20])
        res = random_crop_size_generator(
            batch_size=8, size=(100, 100),
            scale=torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            ratio=torch.tensor([0.9, 1.1], device=device, dtype=dtype),
            same_on_batch=True)
        expected = dict(
            size=torch.tensor([
                [87, 89],
                [87, 89],
                [87, 89],
                [87, 89],
                [87, 89],
                [87, 89],
                [87, 89],
                [87, 89]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['size'], expected['size'])


class TestRandomRectangleGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('height', [200])
    @pytest.mark.parametrize('width', [300])
    @pytest.mark.parametrize('scale', [torch.tensor([.7, 1.1])])
    @pytest.mark.parametrize('ratio', [torch.tensor([.7, 1.1])])
    @pytest.mark.parametrize('value', [0])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_size, height, width, scale, ratio, value, same_on_batch, device, dtype
    ):
        random_rectangles_params_generator(
            batch_size=batch_size, height=height, width=width, scale=scale.to(device=device, dtype=dtype),
            ratio=ratio.to(device=device, dtype=dtype), value=value, same_on_batch=same_on_batch)

    @pytest.mark.parametrize('height,width,scale,ratio,value', [
        pytest.param(-100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3]), 0, marks=pytest.mark.xfail),
        pytest.param(100, -100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3]), 0, marks=pytest.mark.xfail),
        pytest.param(100, -100, torch.tensor([0.7]), torch.tensor([0.7, 1.3]), 0, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([0.7, 1.3, 1.5]), torch.tensor([0.7, 1.3]), 0, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7]), 0, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3, 1.5]), 0, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3]), -1, marks=pytest.mark.xfail),
        pytest.param(100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3]), 2, marks=pytest.mark.xfail),
        pytest.param(
            100, 100, torch.tensor([.5, .7]), torch.tensor([.7, .9]), torch.tensor(0.5), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(
        self, height, width, scale, ratio, value, device, dtype
    ):
        batch_size = 8
        random_rectangles_params_generator(
            batch_size=batch_size, height=height, width=width, scale=scale.to(device=device, dtype=dtype),
            ratio=ratio.to(device=device, dtype=dtype), value=value, same_on_batch=same_on_batch)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        width, height = 100, 150
        scale = torch.tensor([0.7, 1.3], device=device, dtype=dtype)
        ratio = torch.tensor([0.7, 1.3], device=device, dtype=dtype)
        value = 0.5
        res = random_rectangles_params_generator(
            batch_size=2, height=height, width=width, scale=scale,
            ratio=ratio, value=value, same_on_batch=False)
        expected = dict(
            widths=torch.tensor([100, 100], device=device, dtype=torch.int32),
            heights=torch.tensor([0, 0], device=device, dtype=torch.int32),
            xs=torch.tensor([0, 0], device=device, dtype=torch.int32),
            ys=torch.tensor([9, 23], device=device, dtype=torch.int32),
            values=torch.tensor([0.5000, 0.5000], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['widths'], expected['widths'])
        assert_allclose(res['widths'], expected['widths'])
        assert_allclose(res['xs'], expected['xs'])
        assert_allclose(res['ys'], expected['ys'])
        assert_allclose(res['values'], expected['values'])

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        width, height = 100, 150
        scale = torch.tensor([0.7, 1.3], device=device, dtype=dtype)
        ratio = torch.tensor([0.7, 1.3], device=device, dtype=dtype)
        value = 0.5
        res = random_rectangles_params_generator(
            batch_size=2, height=height, width=width, scale=scale,
            ratio=ratio, value=value, same_on_batch=True)
        expected = dict(
            widths=torch.tensor([100, 100], device=device, dtype=torch.int32),
            heights=torch.tensor([0, 0], device=device, dtype=torch.int32),
            xs=torch.tensor([0, 0], device=device, dtype=torch.int32),
            ys=torch.tensor([20, 20], device=device, dtype=torch.int32),
            values=torch.tensor([0.5000, 0.5000], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['widths'], expected['widths'])
        assert_allclose(res['widths'], expected['widths'])
        assert_allclose(res['xs'], expected['xs'])
        assert_allclose(res['ys'], expected['ys'])
        assert_allclose(res['values'], expected['values'])


class TestCenterCropGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [2])
    @pytest.mark.parametrize('height', [200])
    @pytest.mark.parametrize('width', [200])
    @pytest.mark.parametrize('size', [(100, 100)])
    def test_valid_param_combinations(
        self, batch_size, height, width, size, device, dtype
    ):
        center_crop_generator(batch_size=batch_size, height=height, width=width, size=size)

    @pytest.mark.parametrize('height,width,size', [
        pytest.param(200, -200, (100, 100), marks=pytest.mark.xfail),
        pytest.param(-200, 200, (100, 100), marks=pytest.mark.xfail),
        pytest.param(100, 100, (120, 120), marks=pytest.mark.xfail),
        pytest.param(150, 100, (120, 120), marks=pytest.mark.xfail),
        pytest.param(100, 150, (120, 120), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, height, width, size, device, dtype):
        batch_size = 2
        center_crop_generator(batch_size=batch_size, height=height, width=width, size=size)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        res = center_crop_generator(batch_size=2, height=200, width=200, size=(120, 150))
        expected = dict(
            src=torch.tensor([
                [[25, 40],
                 [174, 40],
                 [174, 159],
                 [25, 159]],
                [[25, 40],
                 [174, 40],
                 [174, 159],
                 [25, 159]]], device=device, dtype=torch.long),
            dst=torch.tensor([
                [[0, 0],
                 [149, 0],
                 [149, 119],
                 [0, 119]],
                [[0, 0],
                 [149, 0],
                 [149, 119],
                 [0, 119]]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['src'], expected['src'])
        assert_allclose(res['dst'], expected['dst'])

    def test_same_on_batch(self, device, dtype):
        pass


class TestRandomMotionBlur(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('kernel_size', [1, (3, 5)])
    @pytest.mark.parametrize('angle', [torch.tensor([10, 30])])
    @pytest.mark.parametrize('direction', [torch.tensor([-1, -1]), torch.tensor([1, 1])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, kernel_size, angle, direction, same_on_batch, device, dtype):
        random_motion_blur_generator(
            batch_size=batch_size, kernel_size=kernel_size,
            angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype),
            same_on_batch=same_on_batch)

    @pytest.mark.parametrize('kernel_size,angle,direction', [
        pytest.param(4, torch.tensor([30, 100]), torch.tensor([-1, 1]), marks=pytest.mark.xfail),
        pytest.param(1, torch.tensor([30, 100]), torch.tensor([-1, 1]), marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), torch.tensor([30, 100]), torch.tensor([-1, 1]), marks=pytest.mark.xfail),
        pytest.param(3, torch.tensor([30, 100]), torch.tensor([-2, 1]), marks=pytest.mark.xfail),
        pytest.param(3, torch.tensor([30, 100]), torch.tensor([-1, 2]), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, kernel_size, angle, direction, device, dtype):
        random_motion_blur_generator(
            batch_size=8, kernel_size=kernel_size, angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        angle = torch.tensor([30, 90])
        direction = torch.tensor([-1, 1])
        res = random_motion_blur_generator(
            batch_size=2, kernel_size=3, angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype), same_on_batch=False)
        expected = dict(
            ksize_factor=torch.tensor([3., 3.], device=device, dtype=torch.int32),
            angle_factor=torch.tensor([33.4893, 33.7746], device=device, dtype=dtype),
            direction_factor=torch.tensor([-0.7528, -0.8948], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        angle = torch.tensor([30, 90])
        direction = torch.tensor([-1, 1])
        res = random_motion_blur_generator(
            batch_size=2, kernel_size=3, angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            ksize_factor=torch.tensor([3., 3.], device=device, dtype=torch.int32),
            angle_factor=torch.tensor([33.4893, 33.4893], device=device, dtype=dtype),
            direction_factor=torch.tensor([-0.8742, -0.8742], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)


class TestRandomSolarizeGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('thresholds', [torch.tensor([0, 1]), torch.tensor([0.4, 0.6])])
    @pytest.mark.parametrize('additions', [torch.tensor([-0.5, 0.5])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, thresholds, additions, same_on_batch, device, dtype):
        random_solarize_generator(
            batch_size=batch_size, thresholds=thresholds.to(device=device, dtype=dtype),
            additions=additions.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    @pytest.mark.parametrize('thresholds,additions', [
        pytest.param(torch.tensor([0, 2]), torch.tensor([-0.5, 0.5]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([-1, 1]), torch.tensor([-0.5, 0.5]), marks=pytest.mark.xfail),
        pytest.param([0, 1], torch.tensor([-0.5, 0.5]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([0, 1]), torch.tensor([-0.5, 1]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([0, 1]), torch.tensor([-1, 0.5]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([0, 1]), [-0.5, 0.5], marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, thresholds, additions, device, dtype):
        random_solarize_generator(
            batch_size=batch_size, thresholds=thresholds.to(device=device, dtype=dtype),
            additions=additions.to(device=device, dtype=dtype))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = random_solarize_generator(
            batch_size=batch_size, thresholds=torch.tensor([0, 1], device=device, dtype=dtype),
            additions=torch.tensor([-0.5, 0.5], device=device, dtype=dtype), same_on_batch=False)
        expected = dict(
            thresholds_factor=torch.tensor(
                [0.0582, 0.0629, 0.1236, 0.0526, 0.5262, 0.4768, 0.9552, 0.9288], device=device, dtype=dtype),
            additions_factor=torch.tensor(
                [-0.4165, -0.3674, -0.3429, -0.1246, 0.3425, 0.3705, -0.1225, 0.1124], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['thresholds_factor'], expected['thresholds_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['additions_factor'], expected['additions_factor'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = random_solarize_generator(
            batch_size=batch_size, thresholds=torch.tensor([0, 1], device=device, dtype=dtype),
            additions=torch.tensor([-0.5, 0.5], device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            thresholds_factor=torch.tensor(
                [0.0582, 0.0582, 0.0582, 0.0582, 0.0582, 0.0582, 0.0582, 0.0582], device=device, dtype=dtype),
            additions_factor=torch.tensor(
                [-0.4371, -0.4371, -0.4371, -0.4371, -0.4371, -0.4371, -0.4371, -0.4371], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['thresholds_factor'], expected['thresholds_factor'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['additions_factor'], expected['additions_factor'], rtol=1e-4, atol=1e-4)


class TestRandomPosterizeGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('bits', [torch.tensor([0, 8])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, bits, same_on_batch, device, dtype):
        random_posterize_generator(
            batch_size=batch_size, bits=bits.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    @pytest.mark.parametrize('bits', [
        pytest.param(torch.tensor([-1, 1]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([0, 9]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([3]), marks=pytest.mark.xfail),
        pytest.param([0, 8], marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, bits, device, dtype):
        random_posterize_generator(
            batch_size=batch_size, bits=bits.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(9)
        batch_size = 8
        res = random_posterize_generator(
            batch_size=batch_size, bits=torch.tensor([0, 8], device=device, dtype=dtype), same_on_batch=False)
        expected = dict(
            bits_factor=torch.tensor(
                [1, 6, 2, 0, 0, 4, 0, 0], device=device, dtype=torch.int32)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['bits_factor'], expected['bits_factor'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(9)
        batch_size = 8
        res = random_posterize_generator(
            batch_size=batch_size, bits=torch.tensor([0, 8], device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            bits_factor=torch.tensor(
                [1, 1, 1, 1, 1, 1, 1, 1], device=device, dtype=torch.int32)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['bits_factor'], expected['bits_factor'], rtol=1e-4, atol=1e-4)


class TestRandomSharpnessGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('sharpness', [torch.tensor([0., 1.])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, sharpness, same_on_batch, device, dtype):
        random_sharpness_generator(
            batch_size=batch_size, sharpness=sharpness.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    @pytest.mark.parametrize('sharpness', [
        pytest.param(torch.tensor([-1, 5]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([3]), marks=pytest.mark.xfail),
        pytest.param([0, 1.], marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, sharpness, device, dtype):
        random_sharpness_generator(
            batch_size=batch_size, sharpness=sharpness.to(device=device, dtype=dtype), same_on_batch=same_on_batch)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = random_sharpness_generator(
            batch_size=batch_size, sharpness=torch.tensor([0., 1.], device=device, dtype=dtype), same_on_batch=False)
        expected = dict(
            sharpness_factor=torch.tensor(
                [0.0582, 0.0629, 0.1236, 0.0526, 0.5262, 0.4768, 0.9552, 0.9288], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['sharpness_factor'], expected['sharpness_factor'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = random_sharpness_generator(
            batch_size=batch_size, sharpness=torch.tensor([0., 1.], device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            sharpness_factor=torch.tensor(
                [0.0582, 0.0582, 0.0582, 0.0582, 0.0582, 0.0582, 0.0582, 0.0582], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['sharpness_factor'], expected['sharpness_factor'], rtol=1e-4, atol=1e-4)


class TestRandomMixUpGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('p', [0., 0.5, 1.])
    @pytest.mark.parametrize('lambda_val', [None, torch.tensor([0., 1.])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, p, lambda_val, same_on_batch, device, dtype):
        random_mixup_generator(
            batch_size=batch_size, p=p,
            lambda_val=lambda_val.to(
                device=device, dtype=dtype) if isinstance(lambda_val, (torch.Tensor)) else lambda_val,
            same_on_batch=same_on_batch)

    @pytest.mark.parametrize('lambda_val', [
        pytest.param(torch.tensor([-1, 1]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([0, 2]), marks=pytest.mark.xfail),
        pytest.param(torch.tensor([0, 0.5, 1]), marks=pytest.mark.xfail),
        pytest.param([0., 1.], marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, lambda_val, device, dtype):
        random_mixup_generator(
            batch_size=8, lambda_val=lambda_val.to(device=device, dtype=dtype))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = random_mixup_generator(
            batch_size=batch_size, p=0.5,
            lambda_val=torch.tensor([0., 1.], device=device, dtype=dtype), same_on_batch=False)
        expected = dict(
            mixup_pairs=torch.tensor([6, 1, 0, 7, 2, 5, 3, 4], device=device, dtype=torch.long),
            mixup_lambdas=torch.tensor(
                [0.0000, 0.0000, 0.0196, 0.0000, 0.6535, 0.0000, 0.5949, 0.0000], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['mixup_pairs'], expected['mixup_pairs'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['mixup_lambdas'], expected['mixup_lambdas'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(9)
        batch_size = 8
        res = random_mixup_generator(
            batch_size=batch_size, p=.9999,
            lambda_val=torch.tensor([0., 1.], device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            mixup_pairs=torch.tensor([4, 6, 7, 5, 0, 1, 3, 2], device=device, dtype=torch.long),
            mixup_lambdas=torch.tensor(
                [0.0435, 0.0435, 0.0435, 0.0435, 0.0435, 0.0435, 0.0435, 0.0435], device=device, dtype=dtype)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['mixup_pairs'], expected['mixup_pairs'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['mixup_lambdas'], expected['mixup_lambdas'], rtol=1e-4, atol=1e-4)


class TestRandomCutMixGen(RandomGeneratorBaseTests):

    @pytest.mark.parametrize('batch_size', [1, 8])
    @pytest.mark.parametrize('p', [0, 0.5, 1.])
    @pytest.mark.parametrize('width,height', [(200, 200)])
    @pytest.mark.parametrize('num_mix', [1, 3])
    @pytest.mark.parametrize('beta', [None, torch.tensor(0.), torch.tensor(1.)])
    @pytest.mark.parametrize('cut_size', [None, torch.tensor([0., 1.]), torch.tensor([0.3, 0.6])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_size, p, width, height, num_mix, beta, cut_size, same_on_batch, device, dtype
    ):
        random_cutmix_generator(
            batch_size=batch_size, p=p, width=width, height=height,
            beta=beta.to(device=device, dtype=dtype) if isinstance(beta, (torch.Tensor)) else beta,
            cut_size=cut_size.to(device=device, dtype=dtype) if isinstance(cut_size, (torch.Tensor)) else cut_size,
            same_on_batch=same_on_batch)

    @pytest.mark.parametrize('width,height,num_mix,beta,cut_size', [
        pytest.param(200, -200, 1, None, None, marks=pytest.mark.xfail),
        pytest.param(-200, 200, 1, None, None, marks=pytest.mark.xfail),
        pytest.param(200, 200, 0, None, None, marks=pytest.mark.xfail),
        pytest.param(200, 200, 1.5, None, None, marks=pytest.mark.xfail),
        pytest.param(200, 200, 1, torch.tensor([0., 1.]), None, marks=pytest.mark.xfail),
        pytest.param(200, 200, 1, None, torch.tensor([-1., 1.]), marks=pytest.mark.xfail),
        pytest.param(200, 200, 1, None, torch.tensor([0., 2.]), marks=pytest.mark.xfail),
    ])
    def test_invalid_param_combinations(self, width, height, num_mix, beta, cut_size, device, dtype):
        random_cutmix_generator(
            batch_size=8, p=p, width=width, height=height,
            beta=beta.to(device=device, dtype=dtype) if isinstance(beta, (torch.Tensor)) else beta,
            cut_size=beta.to(device=device, dtype=dtype) if isinstance(cut_size, (torch.Tensor)) else cut_size,
            same_on_batch=same_on_batch)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        res = random_cutmix_generator(
            batch_size=batch_size, width=200, height=200, p=0.5, num_mix=1,
            beta=torch.tensor(1., device=device, dtype=dtype),
            cut_size=torch.tensor([0., 1.], device=device, dtype=dtype), same_on_batch=False)
        expected = dict(
            mix_pairs=torch.tensor([[0., 1.]], device=device, dtype=torch.long),
            crop_src=torch.tensor([[
                [[174, 122],
                 [173, 122],
                 [173, 121],
                 [174, 121]],
                [[75, 17],
                 [74, 17],
                 [74, 16],
                 [75, 16]]]], device=device, dtype=torch.long)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['mix_pairs'], expected['mix_pairs'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['crop_src'], expected['crop_src'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        res = random_cutmix_generator(
            batch_size=batch_size, width=200, height=200, p=0.5, num_mix=1,
            beta=torch.tensor(1., device=device, dtype=dtype),
            cut_size=torch.tensor([0., 1.], device=device, dtype=dtype), same_on_batch=True)
        expected = dict(
            mix_pairs=torch.tensor([[1., 0.]], device=device, dtype=torch.long),
            crop_src=torch.tensor([[
                [[118, 3],
                 [117, 3],
                 [117, 2],
                 [118, 2]],
                [[118, 3],
                 [117, 3],
                 [117, 2],
                 [118, 2]]]], device=device, dtype=torch.long)
        )
        assert res.keys() == expected.keys()
        assert_allclose(res['mix_pairs'], expected['mix_pairs'], rtol=1e-4, atol=1e-4)
        assert_allclose(res['crop_src'], expected['crop_src'], rtol=1e-4, atol=1e-4)
