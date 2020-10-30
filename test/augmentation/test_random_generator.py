
import pytest
import torch
from torch.testing import assert_allclose

from kornia.augmentation.random_generator import (
    random_prob_generator,
    random_color_jitter_generator
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
    def test_random_gen(self, p, expected):
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
        random_color_jitter_generator(batch_size, brightness, contrast, saturation, hue, same_on_batch)

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
        random_color_jitter_generator(8, brightness, contrast, saturation, hue)

    def test_random_gen(self):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = random_color_jitter_generator(
            batch_size, brightness=torch.tensor([0.8, 1.2]), contrast=torch.tensor([0.7, 1.3]),
            saturation=torch.tensor([0.6, 1.4]), hue=torch.tensor([-0.1, 0.1]))

        expected_jitter_params = {
            'brightness_factor': torch.tensor([1.1529, 1.1660, 0.9531, 1.1837, 0.9562, 1.0404, 0.9026, 1.1175]),
            'contrast_factor': torch.tensor([1.2645, 0.7799, 1.2608, 1.0561, 1.2216, 1.0406, 1.1447, 0.9576]),
            'hue_factor': torch.tensor([0.0771, 0.0148, -0.0467, 0.02549, -0.0461, -0.0117, -0.0406, 0.0663]),
            'saturation_factor': torch.tensor([0.6843, 0.8156, 0.8871, 0.7595, 1.0378, 0.6049, 1.3612, 0.6602]),
            'order': torch.tensor([3, 2, 0, 1])
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
            jitter_params['order'], expected_jitter_params['order'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = random_color_jitter_generator(
            batch_size, brightness=torch.tensor([0.8, 1.2]), contrast=torch.tensor([0.7, 1.3]),
            saturation=torch.tensor([0.6, 1.4]), hue=torch.tensor([-0.1, 0.1]), same_on_batch=True)

        expected_res = {
            'brightness_factor': torch.tensor([1.1529] * batch_size),
            'contrast_factor': torch.tensor([1.2490] * batch_size),
            'hue_factor': torch.tensor([-0.0234] * batch_size),
            'saturation_factor': torch.tensor([1.3674] * batch_size),
            'order': torch.tensor([2, 3, 0, 1])
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
            jitter_params['order'], expected_res['order'], rtol=1e-4, atol=1e-4)
