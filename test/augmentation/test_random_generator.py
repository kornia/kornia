import pytest
import torch
from torch import Tensor

from kornia.augmentation.random_generator import (
    AffineGenerator,
    ColorJiggleGenerator,
    ColorJitterGenerator,
    CropGenerator,
    CutmixGenerator,
    MixupGenerator,
    MotionBlurGenerator,
    PerspectiveGenerator,
    PlainUniformGenerator,
    PosterizeGenerator,
    ProbabilityGenerator,
    RectangleEraseGenerator,
    ResizedCropGenerator,
    CenterCropGenerator,
)
from kornia.testing import assert_close
from kornia.utils._compat import torch_version_ge


class RandomGeneratorBaseTests:
    def test_valid_param_combinations(self, device, dtype):
        raise NotImplementedError

    def test_invalid_param_combinations(self, device, dtype):
        raise NotImplementedError

    def test_random_gen(self, device, dtype):
        raise NotImplementedError

    def test_same_on_batch(self, device, dtype):
        raise NotImplementedError


class TestRandomProbGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('p', [0.0, 0.5, 1.0])
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, p, batch_size, same_on_batch):
        ProbabilityGenerator(p)(torch.Size([batch_size]), same_on_batch)

    @pytest.mark.parametrize(
        'p',
        [
            # Should be failed if p > 1. or p < 0.
            (-1.0),
            (2.0),
        ],
    )
    def test_invalid_param_combinations(self, p):
        with pytest.raises(Exception):
            ProbabilityGenerator(p)(torch.Size([8]))

    @pytest.mark.parametrize(
        'p,expected',
        [(0.0, [False] * 8), (0.5, [False, False, True, False, True, False, True, False]), (1.0, [True] * 8)],
    )
    def test_random_gen(self, p, expected, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = ProbabilityGenerator(p)(torch.Size([batch_size]))
        assert (res['probs'] == torch.tensor(expected)).long().sum() == batch_size

    @pytest.mark.parametrize("seed,expected", [(42, [False] * 8), (0, [True] * 8)])
    def test_same_on_batch(self, seed, expected, device, dtype):
        torch.manual_seed(seed)
        batch_size = 8
        res = ProbabilityGenerator(0.5)(torch.Size([batch_size]), True)
        assert (res['probs'] == torch.tensor(expected)).long().sum() == batch_size


class TestColorJiggleGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('brightness', [None, torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize('contrast', [None, torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize('saturation', [None, torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize('hue', [None, torch.tensor([-0.1, 0.1])])
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, brightness, contrast, saturation, hue, batch_size, same_on_batch, device, dtype
    ):
        ColorJiggleGenerator(
            torch.as_tensor(
                brightness if brightness is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
            ),
            torch.as_tensor(contrast if contrast is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype),
            torch.as_tensor(
                saturation if saturation is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
            ),
            torch.as_tensor(hue if hue is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype),
        )(torch.Size([batch_size]), same_on_batch)

    @pytest.mark.parametrize(
        'brightness,contrast,saturation,hue',
        [
            # Should be failed if value out of bounds or tensor.shape != [1, 2]
            (torch.tensor([-1.0, 2.0]), None, None, None),
            (torch.tensor([0.0, 3.0]), None, None, None),
            (torch.tensor([0.0]), None, None, None),
            (torch.tensor([0.0, 1.0, 2.0]), None, None, None),
            (None, torch.tensor([-1.0, 2.0]), None, None),
            (None, torch.tensor([0.0]), None, None),
            (None, torch.tensor([0.0, 1.0, 2.0]), None, None),
            (None, None, torch.tensor([-1.0, 2.0]), None),
            (None, None, torch.tensor([0.0]), None),
            (None, None, torch.tensor([0.0, 1.0, 2.0]), None),
            (None, None, None, torch.tensor([-1.0, 0.0])),
            (None, None, None, torch.tensor([0, 1.0])),
            (None, None, None, torch.tensor([0.0])),
            (None, None, None, torch.tensor([0.0, 1.0, 2.0])),
        ],
    )
    def test_invalid_param_combinations(self, brightness, contrast, saturation, hue, device, dtype):
        with pytest.raises(Exception):
            ColorJiggleGenerator(
                torch.as_tensor(
                    brightness if brightness is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
                ),
                torch.as_tensor(
                    contrast if contrast is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
                ),
                torch.as_tensor(
                    saturation if saturation is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
                ),
                torch.as_tensor(hue if hue is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype),
            )(torch.Size([8]))

    def test_random_gen(self, device, dtype):
        # TODO(jian): crashes with pytorch 1.10, cuda and fp64
        if torch_version_ge(1, 10) and "cuda" in str(device):
            pytest.skip("AssertionError: Tensor-likes are not close!")
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = ColorJiggleGenerator(
            brightness=torch.tensor([0.8, 1.2], device=device, dtype=dtype),
            contrast=torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            saturation=torch.tensor([0.6, 1.4], device=device, dtype=dtype),
            hue=torch.tensor([-0.1, 0.1], device=device, dtype=dtype),
        )(torch.Size([batch_size]))

        expected_jitter_params = {
            'brightness_factor': torch.tensor(
                [1.1529, 1.1660, 0.9531, 1.1837, 0.9562, 1.0404, 0.9026, 1.1175], device=device, dtype=dtype
            ),
            'contrast_factor': torch.tensor(
                [1.2645, 0.7799, 1.2608, 1.0561, 1.2216, 1.0406, 1.1447, 0.9576], device=device, dtype=dtype
            ),
            'hue_factor': torch.tensor(
                [0.0771, 0.0148, -0.0467, 0.0255, -0.0461, -0.0117, -0.0406, 0.0663], device=device, dtype=dtype
            ),
            'saturation_factor': torch.tensor(
                [0.6843, 0.8156, 0.8871, 0.7595, 1.0378, 0.6049, 1.3612, 0.6602], device=device, dtype=dtype
            ),
            'order': torch.tensor([3, 2, 0, 1], device=device, dtype=dtype),
        }

        assert set(list(jitter_params.keys())) == {
            'brightness_factor',
            'contrast_factor',
            'hue_factor',
            'saturation_factor',
            'order',
        }, "Redundant keys found apart from \
                'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor', 'order'"

        assert_close(
            jitter_params['brightness_factor'], expected_jitter_params['brightness_factor'], rtol=1e-4, atol=1e-4
        )
        assert_close(jitter_params['contrast_factor'], expected_jitter_params['contrast_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['hue_factor'], expected_jitter_params['hue_factor'], rtol=1e-4, atol=1e-4)
        assert_close(
            jitter_params['saturation_factor'], expected_jitter_params['saturation_factor'], rtol=1e-4, atol=1e-4
        )
        assert_close(jitter_params['order'].to(dtype), expected_jitter_params['order'], rtol=1e-4, atol=1e-4)

    def test_random_gen_accumulative_additive_additive(self, device, dtype):
        # TODO(jian): crashes with pytorch 1.10, cuda and fp64
        if torch_version_ge(1, 10) and "cuda" in str(device):
            pytest.skip("AssertionError: Tensor-likes are not close!")
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = ColorJiggleGenerator(
            brightness=torch.tensor([0.8, 1.2], device=device, dtype=dtype),
            contrast=torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            saturation=torch.tensor([0.6, 1.4], device=device, dtype=dtype),
            hue=torch.tensor([-0.1, 0.1], device=device, dtype=dtype),
        )(torch.Size([batch_size]))

        expected_jitter_params = {
            'brightness_factor': torch.tensor(
                [1.1529, 1.1660, 0.9531, 1.1837, 0.9562, 1.0404, 0.9026, 1.1175], device=device, dtype=dtype
            ),
            'contrast_factor': torch.tensor(
                [1.2645, 0.7799, 1.2608, 1.0561, 1.2216, 1.0406, 1.1447, 0.9576], device=device, dtype=dtype
            ),
            'hue_factor': torch.tensor(
                [0.0771, 0.0148, -0.0467, 0.0255, -0.0461, -0.0117, -0.0406, 0.0663], device=device, dtype=dtype
            ),
            'saturation_factor': torch.tensor(
                [0.6843, 0.8156, 0.8871, 0.7595, 1.0378, 0.6049, 1.3612, 0.6602], device=device, dtype=dtype
            ),
            'order': torch.tensor([3, 2, 0, 1], device=device, dtype=dtype),
        }

        assert set(list(jitter_params.keys())) == {
            'brightness_factor',
            'contrast_factor',
            'hue_factor',
            'saturation_factor',
            'order',
        }, "Redundant keys found apart from \
                'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor', 'order'"

        assert_close(
            jitter_params['brightness_factor'], expected_jitter_params['brightness_factor'], rtol=1e-4, atol=1e-4
        )
        assert_close(jitter_params['contrast_factor'], expected_jitter_params['contrast_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['hue_factor'], expected_jitter_params['hue_factor'], rtol=1e-4, atol=1e-4)
        assert_close(
            jitter_params['saturation_factor'], expected_jitter_params['saturation_factor'], rtol=1e-4, atol=1e-4
        )
        assert_close(jitter_params['order'].to(dtype), expected_jitter_params['order'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = ColorJiggleGenerator(
            brightness=torch.tensor([0.8, 1.2], device=device, dtype=dtype),
            contrast=torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            saturation=torch.tensor([0.6, 1.4], device=device, dtype=dtype),
            hue=torch.tensor([-0.1, 0.1], device=device, dtype=dtype),
        )(torch.Size([batch_size]), same_on_batch=True)

        expected_res = {
            'brightness_factor': torch.tensor([1.1529] * batch_size, device=device, dtype=dtype),
            'contrast_factor': torch.tensor([1.2490] * batch_size, device=device, dtype=dtype),
            'hue_factor': torch.tensor([-0.0234] * batch_size, device=device, dtype=dtype),
            'saturation_factor': torch.tensor([1.3674] * batch_size, device=device, dtype=dtype),
            'order': torch.tensor([2, 3, 0, 1], device=device, dtype=dtype),
        }

        assert_close(jitter_params['brightness_factor'], expected_res['brightness_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['contrast_factor'], expected_res['contrast_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['hue_factor'], expected_res['hue_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['saturation_factor'], expected_res['saturation_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['order'].to(dtype), expected_res['order'], rtol=1e-4, atol=1e-4)


class TestColorJitterGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('brightness', [None, torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize('contrast', [None, torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize('saturation', [None, torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize('hue', [None, torch.tensor([-0.1, 0.1])])
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, brightness, contrast, saturation, hue, batch_size, same_on_batch, device, dtype
    ):
        ColorJitterGenerator(
            torch.as_tensor(
                brightness if brightness is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
            ),
            torch.as_tensor(contrast if contrast is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype),
            torch.as_tensor(
                saturation if saturation is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
            ),
            torch.as_tensor(hue if hue is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype),
        )(torch.Size([batch_size]), same_on_batch)

    @pytest.mark.parametrize(
        'brightness,contrast,saturation,hue',
        [
            # Should be failed if value out of bounds or tensor.shape != [1, 2]
            (torch.tensor([-1.0, 2.0]), None, None, None),
            (torch.tensor([0.0]), None, None, None),
            (torch.tensor([0.0, 1.0, 2.0]), None, None, None),
            (None, torch.tensor([-1.0, 2.0]), None, None),
            (None, torch.tensor([0.0]), None, None),
            (None, torch.tensor([0.0, 1.0, 2.0]), None, None),
            (None, None, torch.tensor([-1.0, 2.0]), None),
            (None, None, torch.tensor([0.0]), None),
            (None, None, torch.tensor([0.0, 1.0, 2.0]), None),
            (None, None, None, torch.tensor([-1.0, 0.0])),
            (None, None, None, torch.tensor([0, 1.0])),
            (None, None, None, torch.tensor([0.0])),
            (None, None, None, torch.tensor([0.0, 1.0, 2.0])),
        ],
    )
    def test_invalid_param_combinations(self, brightness, contrast, saturation, hue, device, dtype):
        with pytest.raises(Exception):
            ColorJitterGenerator(
                torch.as_tensor(
                    brightness if brightness is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
                ),
                torch.as_tensor(
                    contrast if contrast is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
                ),
                torch.as_tensor(
                    saturation if saturation is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype
                ),
                torch.as_tensor(hue if hue is not None else torch.tensor([0.0, 0.0]), device=device, dtype=dtype),
            )(torch.Size([8]))

    def test_random_gen(self, device, dtype):
        # TODO(jian): crashes with pytorch 1.10, cuda and fp64
        if torch_version_ge(1, 10) and "cuda" in str(device):
            pytest.skip("AssertionError: Tensor-likes are not close!")
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = ColorJitterGenerator(
            brightness=torch.tensor([0.8, 1.2], device=device, dtype=dtype),
            contrast=torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            saturation=torch.tensor([0.6, 1.4], device=device, dtype=dtype),
            hue=torch.tensor([-0.1, 0.1], device=device, dtype=dtype),
        )(torch.Size([batch_size]))

        expected_jitter_params = {
            'brightness_factor': torch.tensor(
                [1.1529, 1.1660, 0.9531, 1.1837, 0.9562, 1.0404, 0.9026, 1.1175], device=device, dtype=dtype
            ),
            'contrast_factor': torch.tensor(
                [1.2645, 0.7799, 1.2608, 1.0561, 1.2216, 1.0406, 1.1447, 0.9576], device=device, dtype=dtype
            ),
            'hue_factor': torch.tensor(
                [0.0771, 0.0148, -0.0467, 0.0255, -0.0461, -0.0117, -0.0406, 0.0663], device=device, dtype=dtype
            ),
            'saturation_factor': torch.tensor(
                [0.6843, 0.8156, 0.8871, 0.7595, 1.0378, 0.6049, 1.3612, 0.6602], device=device, dtype=dtype
            ),
            'order': torch.tensor([3, 2, 0, 1], device=device, dtype=dtype),
        }

        assert set(list(jitter_params.keys())) == {
            'brightness_factor',
            'contrast_factor',
            'hue_factor',
            'saturation_factor',
            'order',
        }, "Redundant keys found apart from \
                'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor', 'order'"

        assert_close(
            jitter_params['brightness_factor'], expected_jitter_params['brightness_factor'], rtol=1e-4, atol=1e-4
        )
        assert_close(jitter_params['contrast_factor'], expected_jitter_params['contrast_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['hue_factor'], expected_jitter_params['hue_factor'], rtol=1e-4, atol=1e-4)
        assert_close(
            jitter_params['saturation_factor'], expected_jitter_params['saturation_factor'], rtol=1e-4, atol=1e-4
        )
        assert_close(jitter_params['order'].to(dtype), expected_jitter_params['order'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = ColorJitterGenerator(
            brightness=torch.tensor([0.8, 1.2], device=device, dtype=dtype),
            contrast=torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            saturation=torch.tensor([0.6, 1.4], device=device, dtype=dtype),
            hue=torch.tensor([-0.1, 0.1], device=device, dtype=dtype),
        )(torch.Size([batch_size]), same_on_batch=True)

        expected_res = {
            'brightness_factor': torch.tensor([1.1529] * batch_size, device=device, dtype=dtype),
            'contrast_factor': torch.tensor([1.2490] * batch_size, device=device, dtype=dtype),
            'hue_factor': torch.tensor([-0.0234] * batch_size, device=device, dtype=dtype),
            'saturation_factor': torch.tensor([1.3674] * batch_size, device=device, dtype=dtype),
            'order': torch.tensor([2, 3, 0, 1], device=device, dtype=dtype),
        }

        assert_close(jitter_params['brightness_factor'], expected_res['brightness_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['contrast_factor'], expected_res['contrast_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['hue_factor'], expected_res['hue_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['saturation_factor'], expected_res['saturation_factor'], rtol=1e-4, atol=1e-4)
        assert_close(jitter_params['order'].to(dtype), expected_res['order'], rtol=1e-4, atol=1e-4)


class TestRandomPerspectiveGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('height,width', [(200, 200)])
    @pytest.mark.parametrize('distortion_scale', [torch.tensor(0.0), torch.tensor(0.5), torch.tensor(1.0)])
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, height, width, distortion_scale, batch_size, same_on_batch, device, dtype):
        PerspectiveGenerator(distortion_scale.to(device=device, dtype=dtype))(
            torch.Size([batch_size, 1, height, width]), same_on_batch
        )

    @pytest.mark.parametrize(
        'height,width,distortion_scale',
        [
            # Should be failed if distortion_scale > 1. or distortion_scale < 0.
            (-100, 100, torch.tensor(0.5)),
            (100, -100, torch.tensor(0.5)),
            (100, 100, torch.tensor(-0.5)),
            (100, 100, torch.tensor(1.5)),
            (100, 100, torch.tensor([0.0, 0.5])),
        ],
    )
    def test_invalid_param_combinations(self, height, width, distortion_scale, device, dtype):
        with pytest.raises(Exception):
            PerspectiveGenerator(distortion_scale.to(device=device, dtype=dtype))(torch.Size([8, 1, height, width]))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        res = PerspectiveGenerator(torch.tensor(0.5, device=device, dtype=dtype))(torch.Size([batch_size, 1, 200, 200]))

        expected = dict(
            start_points=torch.tensor(
                [
                    [[0.0, 0.0], [199.0, 0.0], [199.0, 199.0], [0.0, 199.0]],
                    [[0.0, 0.0], [199.0, 0.0], [199.0, 199.0], [0.0, 199.0]],
                ],
                device=device,
                dtype=dtype,
            ),
            end_points=torch.tensor(
                [
                    [[44.1135, 45.7502], [179.8568, 47.9653], [179.4776, 168.9552], [12.8286, 159.3179]],
                    [[47.0386, 6.6593], [152.2701, 29.6790], [155.5298, 170.6142], [37.0547, 177.5298]],
                ],
                device=device,
                dtype=dtype,
            ),
        )
        assert res.keys() == expected.keys()
        assert_close(res['start_points'], expected['start_points'])
        assert_close(res['end_points'], expected['end_points'])

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        res = PerspectiveGenerator(torch.tensor(0.5, device=device, dtype=dtype))(
            torch.Size([batch_size, 1, 200, 200]), same_on_batch=True
        )
        expected = dict(
            start_points=torch.tensor(
                [[[0.0, 0.0], [199.0, 0.0], [199.0, 199.0], [0.0, 199.0]]], device=device, dtype=dtype
            ).repeat(2, 1, 1),
            end_points=torch.tensor(
                [[[44.1135, 45.7502], [179.8568, 47.9653], [179.4776, 168.9552], [12.8286, 159.3179]]],
                device=device,
                dtype=dtype,
            ).repeat(2, 1, 1),
        )
        assert res.keys() == expected.keys()
        assert_close(res['start_points'], expected['start_points'])
        assert_close(res['end_points'], expected['end_points'])

    def test_sampling_method(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 2
        res = PerspectiveGenerator(torch.tensor(0.5, device=device, dtype=dtype), sampling_method="area_preserving")(
            torch.Size([batch_size, 1, 200, 200])
        )

        expected = dict(
            start_points=torch.tensor(
                [
                    [[0.0, 0.0], [199.0, 0.0], [199.0, 199.0], [0.0, 199.0]],
                    [[0.0, 0.0], [199.0, 0.0], [199.0, 199.0], [0.0, 199.0]],
                ],
                device=device,
                dtype=dtype,
            ),
            end_points=torch.tensor(
                [
                    [[38.2269, 41.5004], [187.2864, 45.9306], [188.0448, 209.0895], [-24.3428, 228.3641]],
                    [[44.0771, -36.6814], [242.4598, 9.3580], [235.9404, 205.7715], [24.1094, 191.9404]],
                ],
                device=device,
                dtype=dtype,
            ),
        )
        assert res.keys() == expected.keys()
        assert_close(res['start_points'], expected['start_points'])
        assert_close(res['end_points'], expected['end_points'])

    def test_not_implemented_sampling_method(self, device, dtype):
        batch_size = 2
        with pytest.raises(NotImplementedError):
            PerspectiveGenerator(torch.tensor(0.5, device=device, dtype=dtype), sampling_method="non_existing_method")(
                torch.Size([batch_size, 1, 200, 200])
            )


class TestRandomAffineGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 4])
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
        AffineGenerator(
            degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shear=shear.to(device=device, dtype=dtype) if shear is not None else None,
        )(torch.Size([batch_size, 1, height, width]), same_on_batch)

    @pytest.mark.parametrize(
        'height,width,degrees,translate,scale,shear',
        [
            (-100, 100, torch.tensor([10, 20]), None, None, None),
            (100, -100, torch.tensor([10, 20]), None, None, None),
            (100, 100, 0.5, None, None, None),
            (100, 100, torch.tensor([10, 20, 30]), None, None, None),
            (100, 100, torch.tensor([10, 20]), torch.tensor([0.1]), None, None),
            (10, 10, torch.tensor([1, 2]), torch.tensor([0.1, 0.2, 0.3]), None, None),
            (100, 100, torch.tensor([10, 20]), None, torch.tensor([1]), None),
            (100, 100, torch.tensor([10, 20]), None, torch.tensor([1, 2, 3]), None),
            (100, 100, torch.tensor([10, 20]), None, None, torch.tensor([1])),
            (10, 10, torch.tensor([1, 2]), None, None, torch.tensor([1, 2, 3])),
            (10, 10, torch.tensor([1, 2]), None, None, torch.tensor([1, 2, 3, 4, 5])),
        ],
    )
    def test_invalid_param_combinations(self, height, width, degrees, translate, scale, shear, device, dtype):
        with pytest.raises(Exception):
            AffineGenerator(
                degrees=degrees.to(device=device, dtype=dtype),
                translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
                scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
                shear=shear.to(device=device, dtype=dtype) if shear is not None else None,
            )(torch.Size([8, 1, height, width]))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20], device=device, dtype=dtype)
        translate = torch.tensor([0.1, 0.1], device=device, dtype=dtype)
        scale = torch.tensor([0.7, 1.2], device=device, dtype=dtype)
        shear = torch.tensor([[10, 20], [10, 20]], device=device, dtype=dtype)
        res = AffineGenerator(
            degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shear=shear.to(device=device, dtype=dtype) if shear is not None else None,
        )(torch.Size([2, 1, 200, 200]))
        expected = dict(
            translations=torch.tensor([[-4.3821, -9.7371], [4.0358, 11.7457]], device=device, dtype=dtype),
            center=torch.tensor([[99.5000, 99.5000], [99.5000, 99.5000]], device=device, dtype=dtype),
            scale=torch.tensor([[0.8914, 0.8914], [1.1797, 1.1797]], device=device, dtype=dtype),
            angle=torch.tensor([18.8227, 19.1500], device=device, dtype=dtype),
            shear_x=torch.tensor([19.4077, 11.3319], device=device, dtype=dtype),
            shear_y=torch.tensor([19.3460, 15.9358], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['translations'], expected['translations'], rtol=1e-4, atol=1e-4)
        assert_close(res['center'], expected['center'], rtol=1e-4, atol=1e-4)
        assert_close(res['scale'], expected['scale'], rtol=1e-4, atol=1e-4)
        assert_close(res['angle'], expected['angle'], rtol=1e-4, atol=1e-4)
        assert_close(res['shear_x'], expected['shear_x'], rtol=1e-4, atol=1e-4)
        assert_close(res['shear_y'], expected['shear_y'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        degrees = torch.tensor([10, 20], device=device, dtype=dtype)
        translate = torch.tensor([0.1, 0.1], device=device, dtype=dtype)
        scale = torch.tensor([0.7, 1.2], device=device, dtype=dtype)
        shear = torch.tensor([[10, 20], [10, 20]], device=device, dtype=dtype)
        res = AffineGenerator(
            degrees=degrees.to(device=device, dtype=dtype),
            translate=translate.to(device=device, dtype=dtype) if translate is not None else None,
            scale=scale.to(device=device, dtype=dtype) if scale is not None else None,
            shear=shear.to(device=device, dtype=dtype) if shear is not None else None,
        )(torch.Size([2, 1, 200, 200]), True)
        expected = dict(
            translations=torch.tensor([[-4.6854, 18.3722], [-4.6854, 18.3722]], device=device, dtype=dtype),
            center=torch.tensor([[99.5000, 99.5000], [99.5000, 99.5000]], device=device, dtype=dtype),
            scale=torch.tensor([[1.1575, 1.1575], [1.1575, 1.1575]], device=device, dtype=dtype),
            angle=torch.tensor([18.8227, 18.8227], device=device, dtype=dtype),
            shear_x=torch.tensor([13.9045, 13.9045], device=device, dtype=dtype),
            shear_y=torch.tensor([16.0090, 16.0090], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['translations'], expected['translations'], rtol=1e-4, atol=1e-4)
        assert_close(res['center'], expected['center'], rtol=1e-4, atol=1e-4)
        assert_close(res['scale'], expected['scale'], rtol=1e-4, atol=1e-4)
        assert_close(res['angle'], expected['angle'], rtol=1e-4, atol=1e-4)
        assert_close(res['shear_x'], expected['shear_x'], rtol=1e-4, atol=1e-4)
        assert_close(res['shear_y'], expected['shear_y'], rtol=1e-4, atol=1e-4)


class TestRandomCropGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 2])
    @pytest.mark.parametrize('input_size', [(200, 200)])
    @pytest.mark.parametrize('size', [(100, 100), torch.tensor([50, 50])])
    @pytest.mark.parametrize('resize_to', [None, (100, 100)])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, input_size, size, resize_to, same_on_batch, device, dtype):
        if isinstance(size, Tensor):
            size = size.repeat(batch_size, 1).to(device=device, dtype=dtype)
        CropGenerator(size, resize_to)(torch.Size([batch_size, 1, *input_size]), same_on_batch)

    @pytest.mark.parametrize(
        'input_size,size,resize_to',
        [((-300, 300), (200, 200), (100, 100)), ((200, 200), torch.tensor([50, 50]), (100, 100))],
    )
    def test_invalid_param_combinations(self, input_size, size, resize_to, device, dtype):
        batch_size = 2
        with pytest.raises(Exception):
            CropGenerator(size.to(device=device, dtype=dtype) if isinstance(size, Tensor) else size, resize_to)(
                torch.Size([batch_size, 1, *input_size])
            )

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        res = CropGenerator(torch.tensor([[50, 60], [70, 80]], device=device, dtype=dtype), (200, 200))(
            torch.Size([2, 1, 100, 100])
        )
        expected = dict(
            src=torch.tensor(
                [[[36, 19], [95, 19], [95, 68], [36, 68]], [[19, 29], [98, 29], [98, 98], [19, 98]]],
                device=device,
                dtype=dtype,
            ),
            dst=torch.tensor(
                [[[0, 0], [199, 0], [199, 199], [0, 199]], [[0, 0], [199, 0], [199, 199], [0, 199]]],
                device=device,
                dtype=dtype,
            ),
            input_size=torch.tensor([[100, 100], [100, 100]], device=device, dtype=torch.long),
            output_size=torch.tensor([[200, 200], [200, 200]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_close(res['src'], expected['src'])
        assert_close(res['dst'], expected['dst'])

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        res = CropGenerator(torch.tensor([[50, 60], [70, 80]], device=device, dtype=dtype), (200, 200))(
            torch.Size([2, 1, 100, 100]), True
        )
        expected = dict(
            src=torch.tensor(
                [[[36, 46], [95, 46], [95, 95], [36, 95]], [[36, 46], [115, 46], [115, 115], [36, 115]]],
                device=device,
                dtype=dtype,
            ),
            dst=torch.tensor(
                [[[0, 0], [199, 0], [199, 199], [0, 199]], [[0, 0], [199, 0], [199, 199], [0, 199]]],
                device=device,
                dtype=dtype,
            ),
            input_size=torch.tensor([[100, 100], [100, 100]], device=device, dtype=torch.long),
            output_size=torch.tensor([[200, 200], [200, 200]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_close(res['src'], expected['src'])
        assert_close(res['dst'], expected['dst'])


class TestRandomCropSizeGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('size', [(200, 200)])
    @pytest.mark.parametrize('scale', [torch.tensor([0.7, 1.3])])
    @pytest.mark.parametrize('ratio', [torch.tensor([0.9, 1.1])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, size, scale, ratio, same_on_batch, device, dtype):
        ResizedCropGenerator(
            size, torch.as_tensor(scale, device=device, dtype=dtype), torch.as_tensor(ratio, device=device, dtype=dtype)
        )(torch.Size([batch_size, 1, 300, 300]), same_on_batch)

    @pytest.mark.parametrize(
        'size,scale,ratio',
        [
            ((100), torch.tensor([0.7, 1.3]), torch.tensor([0.9, 1.1])),
            ((100, 100, 100), torch.tensor([0.7, 1.3]), torch.tensor([0.9, 1.1])),
            ((100, 100), torch.tensor([0.7]), torch.tensor([0.9, 1.1])),
            ((100, 100), torch.tensor([0.7, 1.3, 1.5]), torch.tensor([0.9, 1.1])),
            ((100, 100), torch.tensor([0.7, 1.3]), torch.tensor([0.9])),
            ((100, 100), torch.tensor([0.7, 1.3]), torch.tensor([0.9, 1.1, 1.3])),
        ],
    )
    def test_invalid_param_combinations(self, size, scale, ratio, device, dtype):
        batch_size = 2
        with pytest.raises(Exception):
            ResizedCropGenerator(
                size,
                torch.as_tensor(scale, device=device, dtype=dtype),
                torch.as_tensor(ratio, device=device, dtype=dtype),
            )(torch.Size([batch_size, 1, 300, 300]))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        res = ResizedCropGenerator(
            (100, 100),
            torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            torch.tensor([0.9, 1.1], device=device, dtype=dtype),
        )(torch.Size([2, 1, 300, 300]))
        expected = dict(
            src=torch.tensor(
                [
                    [[3.0, 12.0], [298.0, 12.0], [298.0, 294.0], [3.0, 294.0]],
                    [[8.0, 20.0], [284.0, 20.0], [284.0, 298.0], [8.0, 298.0]],
                ],
                device=device,
                dtype=dtype,
            ),
            dst=torch.tensor(
                [
                    [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0]],
                    [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0]],
                ],
                device=device,
                dtype=dtype,
            ),
            input_size=torch.tensor([[300, 300], [300, 300]], device=device, dtype=torch.int64),
            output_size=torch.tensor([[100, 100], [100, 100]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_close(res['src'], expected['src'])
        assert_close(res['dst'], expected['dst'])
        assert_close(res['input_size'], expected['input_size'])

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        res = ResizedCropGenerator(
            (100, 100),
            torch.tensor([0.7, 1.3], device=device, dtype=dtype),
            torch.tensor([0.9, 1.1], device=device, dtype=dtype),
        )(torch.Size([2, 1, 300, 300]), same_on_batch=True)
        expected = dict(
            src=torch.tensor(
                [
                    [[5.0, 0.0], [283.0, 0.0], [283.0, 298.0], [5.0, 298.0]],
                    [[5.0, 0.0], [283.0, 0.0], [283.0, 298.0], [5.0, 298.0]],
                ],
                device=device,
                dtype=dtype,
            ),
            dst=torch.tensor(
                [
                    [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0]],
                    [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0]],
                ],
                device=device,
                dtype=dtype,
            ),
            input_size=torch.tensor([[300, 300], [300, 300]], device=device, dtype=torch.int64),
            output_size=torch.tensor([[100, 100], [100, 100]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_close(res['src'], expected['src'])
        assert_close(res['dst'], expected['dst'])
        assert_close(res['input_size'], expected['input_size'])


class TestRandomRectangleGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('height', [200])
    @pytest.mark.parametrize('width', [300])
    @pytest.mark.parametrize('scale', [torch.tensor([0.7, 1.1])])
    @pytest.mark.parametrize('ratio', [torch.tensor([0.7, 1.1])])
    @pytest.mark.parametrize('value', [0])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_size, height, width, scale, ratio, value, same_on_batch, device, dtype
    ):
        RectangleEraseGenerator(
            scale=scale.to(device=device, dtype=dtype), ratio=ratio.to(device=device, dtype=dtype), value=value
        )(torch.Size([batch_size, 1, height, width]), same_on_batch)

    @pytest.mark.parametrize(
        'height,width,scale,ratio,value',
        [
            (-100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3]), 0),
            (100, -100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3]), 0),
            (100, -100, torch.tensor([0.7]), torch.tensor([0.7, 1.3]), 0),
            (100, 100, torch.tensor([0.7, 1.3, 1.5]), torch.tensor([0.7, 1.3]), 0),
            (100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7]), 0),
            (100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3, 1.5]), 0),
            (100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3]), -1),
            (100, 100, torch.tensor([0.7, 1.3]), torch.tensor([0.7, 1.3]), 2),
            (100, 100, torch.tensor([0.5, 0.7]), torch.tensor([0.7, 0.9]), torch.tensor(0.5)),
        ],
    )
    def test_invalid_param_combinations(self, height, width, scale, ratio, value, device, dtype):
        batch_size = 8
        with pytest.raises(Exception):
            RectangleEraseGenerator(
                scale=scale.to(device=device, dtype=dtype), ratio=ratio.to(device=device, dtype=dtype), value=value
            )(torch.Size([batch_size, 1, height, width]))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        width, height = 100, 150
        scale = torch.tensor([0.7, 1.3], device=device, dtype=dtype)
        ratio = torch.tensor([0.7, 1.3], device=device, dtype=dtype)
        value = 0.5
        res = RectangleEraseGenerator(
            scale=scale.to(device=device, dtype=dtype), ratio=ratio.to(device=device, dtype=dtype), value=value
        )(torch.Size([2, 1, height, width]))
        expected = dict(
            widths=torch.tensor([100, 100], device=device, dtype=dtype),
            heights=torch.tensor([0, 0], device=device, dtype=dtype),
            xs=torch.tensor([0, 0], device=device, dtype=dtype),
            ys=torch.tensor([6, 8], device=device, dtype=dtype),
            values=torch.tensor([0.5000, 0.5000], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['widths'], expected['widths'])
        assert_close(res['widths'], expected['widths'])
        assert_close(res['xs'], expected['xs'])
        assert_close(res['ys'], expected['ys'])
        assert_close(res['values'], expected['values'])

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        width, height = 100, 150
        scale = torch.tensor([0.7, 1.3], device=device, dtype=dtype)
        ratio = torch.tensor([0.7, 1.3], device=device, dtype=dtype)
        value = 0.5
        res = RectangleEraseGenerator(
            scale=scale.to(device=device, dtype=dtype), ratio=ratio.to(device=device, dtype=dtype), value=value
        )(torch.Size([2, 1, height, width]), True)
        expected = dict(
            widths=torch.tensor([100, 100], device=device, dtype=dtype),
            heights=torch.tensor([0, 0], device=device, dtype=dtype),
            xs=torch.tensor([0, 0], device=device, dtype=dtype),
            ys=torch.tensor([10, 10], device=device, dtype=dtype),
            values=torch.tensor([0.5000, 0.5000], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['widths'], expected['widths'])
        assert_close(res['widths'], expected['widths'])
        assert_close(res['xs'], expected['xs'])
        assert_close(res['ys'], expected['ys'])
        assert_close(res['values'], expected['values'])


class TestCenterCropGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 2])
    @pytest.mark.parametrize('height', [200])
    @pytest.mark.parametrize('width', [200])
    @pytest.mark.parametrize('size', [(100, 100)])
    def test_valid_param_combinations(self, batch_size, height, width, size, device, dtype):
        CenterCropGenerator(size).to(device=device, dtype=dtype)(torch.Size((batch_size, 3, height, width)))

    @pytest.mark.parametrize(
        'height,width,size',
        [
            (200, -200, (100, 100)),
            (-200, 200, (100, 100)),
            (100, 100, (120, 120)),
            (150, 100, (120, 120)),
            (100, 150, (120, 120)),
        ],
    )
    def test_invalid_param_combinations(self, height, width, size, device, dtype):
        batch_size = 2
        with pytest.raises(Exception):
            CenterCropGenerator(size).to(device=device, dtype=dtype)(torch.Size((batch_size, 3, height, width)))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        res = CenterCropGenerator((120, 150)).to(device=device, dtype=dtype)(torch.Size((2, 3, 200, 200)))
        expected = dict(
            src=torch.tensor(
                [[[25, 40], [174, 40], [174, 159], [25, 159]], [[25, 40], [174, 40], [174, 159], [25, 159]]],
                device=device,
                dtype=torch.long,
            ),
            dst=torch.tensor(
                [[[0, 0], [149, 0], [149, 119], [0, 119]], [[0, 0], [149, 0], [149, 119], [0, 119]]],
                device=device,
                dtype=torch.long,
            ),
            input_size=torch.tensor([[200, 200], [200, 200]], device=device, dtype=torch.long),
            output_size=torch.tensor([[120, 150], [120, 150]], device=device, dtype=torch.long),
        )
        assert res.keys() == expected.keys()
        assert_close(res['src'].to(device=device), expected['src'])
        assert_close(res['dst'].to(device=device), expected['dst'])

    def test_same_on_batch(self, device, dtype):
        pass


class TestRandomMotionBlur(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('kernel_size', [3, (3, 5)])
    @pytest.mark.parametrize('angle', [torch.tensor([10, 30])])
    @pytest.mark.parametrize('direction', [torch.tensor([-1, -1]), torch.tensor([1, 1])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, kernel_size, angle, direction, same_on_batch, device, dtype):
        MotionBlurGenerator(
            kernel_size=kernel_size,
            angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype),
        )(torch.Size([batch_size]), same_on_batch)

    @pytest.mark.parametrize(
        'kernel_size,angle,direction',
        [
            (4, torch.tensor([30, 100]), torch.tensor([-1, 1])),
            (1, torch.tensor([30, 100]), torch.tensor([-1, 1])),
            ((1, 2, 3), torch.tensor([30, 100]), torch.tensor([-1, 1])),
            (3, torch.tensor([30, 100]), torch.tensor([-2, 1])),
            (3, torch.tensor([30, 100]), torch.tensor([-1, 2])),
        ],
    )
    def test_invalid_param_combinations(self, kernel_size, angle, direction, device, dtype):
        with pytest.raises(Exception):
            MotionBlurGenerator(
                kernel_size=kernel_size,
                angle=angle.to(device=device, dtype=dtype),
                direction=direction.to(device=device, dtype=dtype),
            )(torch.Size([8]))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        angle = torch.tensor([30, 90])
        direction = torch.tensor([-1, 1])
        res = MotionBlurGenerator(
            kernel_size=3,
            angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype),
        )(torch.Size([2]))
        expected = dict(
            ksize_factor=torch.tensor([3, 3], device=device, dtype=torch.int32),
            angle_factor=torch.tensor([82.9362, 84.9002], device=device, dtype=dtype),
            direction_factor=torch.tensor([-0.2343, 0.9186], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
        assert_close(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
        assert_close(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        angle = torch.tensor([30, 90])
        direction = torch.tensor([-1, 1])
        res = MotionBlurGenerator(
            kernel_size=3,
            angle=angle.to(device=device, dtype=dtype),
            direction=direction.to(device=device, dtype=dtype),
        )(torch.Size([2]), True)
        expected = dict(
            ksize_factor=torch.tensor([3, 3], device=device, dtype=torch.int32),
            angle_factor=torch.tensor([82.9362, 82.9362], device=device, dtype=dtype),
            direction_factor=torch.tensor([0.8300, 0.8300], device=device, dtype=dtype),
        )
        assert res.keys() == expected.keys()
        assert_close(res['ksize_factor'], expected['ksize_factor'], rtol=1e-4, atol=1e-4)
        assert_close(res['angle_factor'], expected['angle_factor'], rtol=1e-4, atol=1e-4)
        assert_close(res['direction_factor'], expected['direction_factor'], rtol=1e-4, atol=1e-4)


class TestRandomPosterizeGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('bits', [torch.tensor([0, 8])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, bits, same_on_batch, device, dtype):
        PosterizeGenerator(bits)(torch.Size([batch_size]), same_on_batch)

    @pytest.mark.parametrize('bits', [(torch.tensor([-1, 1])), (torch.tensor([0, 9])), (torch.tensor([3])), ([0, 8],)])
    def test_invalid_param_combinations(self, bits, device, dtype):
        with pytest.raises(Exception):
            if isinstance(bits, Tensor):
                PosterizeGenerator(bits.to(device=device, dtype=dtype))(torch.Size([3]))
            else:
                PosterizeGenerator(bits)(torch.Size([3]))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(9)
        batch_size = 8
        res = PosterizeGenerator(bits=torch.tensor([0, 8], device=device, dtype=dtype))(torch.Size([batch_size]))
        expected = dict(bits_factor=torch.tensor([5, 2, 4, 6, 7, 7, 2, 8], device=device, dtype=torch.int32))
        assert res.keys() == expected.keys()
        assert_close(res['bits_factor'], expected['bits_factor'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(9)
        batch_size = 8
        res = PosterizeGenerator(bits=torch.tensor([0, 8], device=device, dtype=dtype))(
            torch.Size([batch_size]), same_on_batch=True
        )
        expected = dict(bits_factor=torch.tensor([5, 5, 5, 5, 5, 5, 5, 5], device=device, dtype=torch.int32))
        assert res.keys() == expected.keys()
        assert_close(res['bits_factor'], expected['bits_factor'], rtol=1e-4, atol=1e-4)


class TestPlainUniformGenerator(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('sharpness', [torch.tensor([0.0, 1.0])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, sharpness, same_on_batch, device, dtype):
        PlainUniformGenerator(
            (sharpness.to(device=device, dtype=dtype), "sharpness", None, None),
            (sharpness.to(device=device, dtype=dtype), "sharpness_xx", 0.0, (0.0, 1.0)),
        )(torch.Size([batch_size, 1]), same_on_batch)

    @pytest.mark.parametrize('sharpness', [(torch.tensor([-1, 5])), (torch.tensor([3])), ([0, 1.0])])
    def test_invalid_param_combinations(self, sharpness, device, dtype):
        with pytest.raises(Exception):
            PlainUniformGenerator(
                (sharpness.to(device=device, dtype=dtype), "sharpness", None, None),
                (sharpness.to(device=device, dtype=dtype), "sharpness", 0.0, (0.0, 1.0)),
            )(torch.Size([8, 1]))

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = PlainUniformGenerator(
            (torch.tensor([0.0, 1.0], device=device, dtype=dtype), "sharpness_factor", None, None)
        )(torch.Size([batch_size, 1]))
        expected = dict(
            sharpness_factor=torch.tensor(
                [0.8823, 0.9150, 0.3829, 0.9593, 0.3904, 0.6009, 0.2566, 0.7936], device=device, dtype=dtype
            )
        )
        assert res.keys() == expected.keys()
        assert_close(res['sharpness_factor'], expected['sharpness_factor'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = PlainUniformGenerator(
            (torch.tensor([0.0, 1.0], device=device, dtype=dtype), "sharpness_factor", None, None)
        )(torch.Size([batch_size, 1]), True)
        expected = dict(
            sharpness_factor=torch.tensor(
                [0.8823, 0.8823, 0.8823, 0.8823, 0.8823, 0.8823, 0.8823, 0.8823], device=device, dtype=dtype
            )
        )
        assert res.keys() == expected.keys()
        assert_close(res['sharpness_factor'], expected['sharpness_factor'], rtol=1e-4, atol=1e-4)


class TestRandomMixUpGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('p', [0.0, 0.5, 1.0])
    @pytest.mark.parametrize('lambda_val', [None, torch.tensor([0.0, 1.0])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(self, batch_size, p, lambda_val, same_on_batch, device, dtype):
        MixupGenerator(
            p=p,
            lambda_val=lambda_val.to(device=device, dtype=dtype) if isinstance(lambda_val, (Tensor)) else lambda_val,
        )(torch.Size([batch_size, 3, 200, 200]), same_on_batch=same_on_batch)

    @pytest.mark.parametrize(
        'lambda_val', [(torch.tensor([-1, 1])), (torch.tensor([0, 2])), (torch.tensor([0, 0.5, 1])), ([0.0, 1.0])]
    )
    def test_invalid_param_combinations(self, lambda_val, device, dtype):
        with pytest.raises(Exception):
            MixupGenerator(p=1.0, lambda_val=lambda_val.to(device=device, dtype=dtype))(
                torch.Size([8, 3, 200, 200]), same_on_batch=False
            )

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        batch_size = 8
        res = MixupGenerator(p=0.5, lambda_val=torch.tensor([0.0, 1.0], device=device, dtype=dtype))(
            torch.Size([batch_size, 3, 200, 200]), same_on_batch=False
        )
        expected = dict(
            mixup_pairs=torch.tensor([6, 1, 0, 7, 2, 5, 3, 4], device=device, dtype=torch.long),
            mixup_lambdas=torch.tensor(
                [0.0000, 0.0000, 0.5739, 0.0000, 0.6274, 0.0000, 0.4414, 0.0000], device=device, dtype=dtype
            ),
        )
        assert res.keys() == expected.keys()
        assert_close(res['mixup_pairs'], expected['mixup_pairs'], rtol=1e-4, atol=1e-4)
        assert_close(res['mixup_lambdas'], expected['mixup_lambdas'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(9)
        batch_size = 8
        res = MixupGenerator(p=0.999999, lambda_val=torch.tensor([0.0, 1.0], device=device, dtype=dtype))(
            torch.Size([batch_size, 3, 200, 200]), same_on_batch=True
        )
        expected = dict(
            mixup_pairs=torch.tensor([4, 6, 7, 5, 0, 1, 3, 2], device=device, dtype=torch.long),
            mixup_lambdas=torch.tensor(
                [0.3804, 0.3804, 0.3804, 0.3804, 0.3804, 0.3804, 0.3804, 0.3804], device=device, dtype=dtype
            ),
        )
        assert res.keys() == expected.keys()
        assert_close(res['mixup_pairs'], expected['mixup_pairs'], rtol=1e-4, atol=1e-4)
        assert_close(res['mixup_lambdas'], expected['mixup_lambdas'], rtol=1e-4, atol=1e-4)


class TestRandomCutMixGen(RandomGeneratorBaseTests):
    @pytest.mark.parametrize('batch_size', [0, 1, 8])
    @pytest.mark.parametrize('p', [0, 0.5, 1.0])
    @pytest.mark.parametrize('width,height', [(200, 200)])
    @pytest.mark.parametrize('num_mix', [1, 3])
    @pytest.mark.parametrize('beta', [None, torch.tensor(1e-15), torch.tensor(1.0)])
    @pytest.mark.parametrize('cut_size', [None, torch.tensor([0.0, 1.0]), torch.tensor([0.3, 0.6])])
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_valid_param_combinations(
        self, batch_size, p, width, height, num_mix, beta, cut_size, same_on_batch, device, dtype
    ):
        CutmixGenerator(
            p=p,
            num_mix=num_mix,
            beta=beta.to(device=device, dtype=dtype) if isinstance(beta, (Tensor)) else beta,
            cut_size=cut_size.to(device=device, dtype=dtype) if isinstance(cut_size, (Tensor)) else cut_size,
        )(torch.Size([batch_size, 3, height, width]), same_on_batch=same_on_batch)

    @pytest.mark.parametrize(
        'width,height,num_mix,beta,cut_size',
        [
            (200, -200, 1, None, None),
            (-200, 200, 1, None, None),
            (200, 200, 0, None, None),
            (200, 200, 1.5, None, None),
            (200, 200, 1, torch.tensor([0.0, 1.0]), None),
            (200, 200, 1, None, torch.tensor([-1.0, 1.0])),
            (200, 200, 1, None, torch.tensor([0.0, 2.0])),
        ],
    )
    @pytest.mark.parametrize('same_on_batch', [True, False])
    def test_invalid_param_combinations(self, width, height, num_mix, beta, cut_size, same_on_batch, device, dtype):
        with pytest.raises(Exception):
            CutmixGenerator(
                p=0.5,
                num_mix=num_mix,
                beta=beta.to(device=device, dtype=dtype) if isinstance(beta, (Tensor)) else beta,
                cut_size=cut_size.to(device=device, dtype=dtype) if isinstance(cut_size, (Tensor)) else cut_size,
            )(torch.Size([8, 3, height, width]), same_on_batch=same_on_batch)

    def test_random_gen(self, device, dtype):
        torch.manual_seed(42)
        image_shape = torch.Size([8, 3, 200, 200])
        res = CutmixGenerator(
            p=0.5,
            num_mix=1,
            beta=torch.tensor(1.0, device=device, dtype=dtype),
            cut_size=torch.tensor([0.0, 1.0], device=device, dtype=dtype),
        )(image_shape, same_on_batch=False)

        expected = dict(
            mix_pairs=torch.tensor([[1, 7, 5, 3, 6, 4, 2, 0]], device=device, dtype=torch.long),
            crop_src=torch.tensor(
                [
                    [
                        [[48.0, 31.0], [47.0, 31.0], [47.0, 30.0], [48.0, 30.0]],
                        [[48.0, 31.0], [47.0, 31.0], [47.0, 30.0], [48.0, 30.0]],
                        [[17.0, 11.0], [141.0, 11.0], [141.0, 135.0], [17.0, 135.0]],
                        [[48.0, 31.0], [47.0, 31.0], [47.0, 30.0], [48.0, 30.0]],
                        [[16.0, 10.0], [147.0, 10.0], [147.0, 141.0], [16.0, 141.0]],
                        [[48.0, 31.0], [47.0, 31.0], [47.0, 30.0], [48.0, 30.0]],
                        [[8.0, 5.0], [171.0, 5.0], [171.0, 168.0], [8.0, 168.0]],
                        [[48.0, 31.0], [47.0, 31.0], [47.0, 30.0], [48.0, 30.0]],
                    ]
                ],
                device=device,
                dtype=dtype,
            ),
            image_shape=image_shape,
        )
        assert res.keys() == expected.keys(), res.keys()
        assert_close(res['mix_pairs'], expected['mix_pairs'], rtol=1e-4, atol=1e-4)
        assert_close(res['crop_src'], expected['crop_src'], rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        torch.manual_seed(42)
        image_shape = torch.Size([2, 3, 200, 200])
        res = CutmixGenerator(
            p=0.5,
            num_mix=1,
            beta=torch.tensor(1.0, device=device, dtype=dtype),
            cut_size=torch.tensor([0.0, 1.0], device=device, dtype=dtype),
        )(image_shape, same_on_batch=True)
        expected = dict(
            mix_pairs=torch.tensor([[1, 0]], device=device, dtype=torch.long),
            crop_src=torch.tensor(
                [[[[114, 53], [113, 53], [113, 52], [114, 52]], [[114, 53], [113, 53], [113, 52], [114, 52]]]],
                device=device,
                dtype=dtype,
            ),
            image_shape=image_shape,
        )
        assert res.keys() == expected.keys(), res.keys()
        assert_close(res['mix_pairs'], expected['mix_pairs'], rtol=1e-4, atol=1e-4)
        assert_close(res['crop_src'], expected['crop_src'], rtol=1e-4, atol=1e-4)
