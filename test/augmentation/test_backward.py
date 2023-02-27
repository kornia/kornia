import pytest
import torch
import torch.nn as nn

from kornia.augmentation import (
    ColorJiggle,
    ColorJitter,
    RandomAffine,
    RandomErasing,
    RandomMotionBlur,
    RandomPerspective,
    RandomResizedCrop,
    RandomRotation,
    RandomSharpness,
)


class TestColorJiggleBackward:
    @pytest.mark.parametrize("brightness", [0.8, torch.tensor(0.8), torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize("contrast", [0.8, torch.tensor(0.8), torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize("saturation", [0.8, torch.tensor(0.8), torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize("hue", [0.1, torch.tensor(0.1), torch.tensor([-0.1, 0.1])])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_param(self, brightness, contrast, saturation, hue, same_on_batch, device, dtype):
        _brightness = (
            brightness
            if isinstance(brightness, (int, float))
            else nn.Parameter(brightness.clone().to(device=device, dtype=dtype))
        )
        _contrast = (
            contrast
            if isinstance(contrast, (int, float))
            else nn.Parameter(contrast.clone().to(device=device, dtype=dtype))
        )
        _saturation = (
            saturation
            if isinstance(saturation, (int, float))
            else nn.Parameter(saturation.clone().to(device=device, dtype=dtype))
        )
        _hue = hue if isinstance(hue, (int, float)) else nn.Parameter(hue.clone().to(device=device, dtype=dtype))

        torch.manual_seed(0)
        input = torch.randint(255, (2, 3, 10, 10), device=device, dtype=dtype) / 255.0
        aug = ColorJiggle(_brightness, _contrast, _saturation, _hue, same_on_batch)

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=10)
            loss = mse(output, torch.ones_like(output) * 2)
            loss.backward()
            opt.step()

        if not isinstance(brightness, (int, float)):
            assert isinstance(aug.brightness, torch.Tensor)
            # Assert if param not updated
            assert (brightness.to(device=device, dtype=dtype) - aug.brightness.data).sum() != 0
        if not isinstance(contrast, (int, float)):
            assert isinstance(aug.contrast, torch.Tensor)
            # Assert if param not updated
            assert (contrast.to(device=device, dtype=dtype) - aug.contrast.data).sum() != 0
        if not isinstance(saturation, (int, float)):
            assert isinstance(aug.saturation, torch.Tensor)
            # Assert if param not updated
            assert (saturation.to(device=device, dtype=dtype) - aug.saturation.data).sum() != 0
        if not isinstance(hue, (int, float)):
            assert isinstance(aug.hue, torch.Tensor)
            # Assert if param not updated
            assert (hue.to(device=device, dtype=dtype) - aug.hue.data).sum() != 0


class TestColorJitterBackward:
    @pytest.mark.parametrize("brightness", [0.8, torch.tensor(0.8), torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize("contrast", [0.8, torch.tensor(0.8), torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize("saturation", [0.8, torch.tensor(0.8), torch.tensor([0.8, 1.2])])
    @pytest.mark.parametrize("hue", [0.1, torch.tensor(0.1), torch.tensor([-0.1, 0.1])])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_param(self, brightness, contrast, saturation, hue, same_on_batch, device, dtype):
        _brightness = (
            brightness
            if isinstance(brightness, (int, float))
            else nn.Parameter(brightness.clone().to(device=device, dtype=dtype))
        )
        _contrast = (
            contrast
            if isinstance(contrast, (int, float))
            else nn.Parameter(contrast.clone().to(device=device, dtype=dtype))
        )
        _saturation = (
            saturation
            if isinstance(saturation, (int, float))
            else nn.Parameter(saturation.clone().to(device=device, dtype=dtype))
        )
        _hue = hue if isinstance(hue, (int, float)) else nn.Parameter(hue.clone().to(device=device, dtype=dtype))

        torch.manual_seed(0)
        input = torch.randint(255, (2, 3, 10, 10), device=device, dtype=dtype) / 255.0
        aug = ColorJitter(_brightness, _contrast, _saturation, _hue, same_on_batch)

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=10)
            loss = mse(output, torch.ones_like(output) * 2)
            loss.backward()
            opt.step()

        if not isinstance(brightness, (int, float)):
            assert isinstance(aug.brightness, torch.Tensor)
            # Assert if param not updated
            assert (brightness.to(device=device, dtype=dtype) - aug.brightness.data).sum() != 0
        if not isinstance(contrast, (int, float)):
            assert isinstance(aug.contrast, torch.Tensor)
            # Assert if param not updated
            assert (contrast.to(device=device, dtype=dtype) - aug.contrast.data).sum() != 0
        if not isinstance(saturation, (int, float)):
            assert isinstance(aug.saturation, torch.Tensor)
            # Assert if param not updated
            assert (saturation.to(device=device, dtype=dtype) - aug.saturation.data).sum() != 0
        if not isinstance(hue, (int, float)):
            assert isinstance(aug.hue, torch.Tensor)
            # Assert if param not updated
            assert (hue.to(device=device, dtype=dtype) - aug.hue.data).sum() != 0


class TestRandomAffineBackward:
    @pytest.mark.parametrize("degrees", [10, [10.0, 20.0], (10, 20), torch.tensor(10.0), torch.tensor([10, 20])])
    @pytest.mark.parametrize("translate", [[0.1, 0.2], torch.tensor([0.1, 0.2])])
    @pytest.mark.parametrize(
        "scale", [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4], torch.tensor([0.1, 0.2]), torch.tensor([0.1, 0.2, 0.3, 0.4])]
    )
    @pytest.mark.parametrize(
        "shear", [[10.0, 20.0], [10.0, 20.0, 30.0, 40.0], torch.tensor([10, 20]), torch.tensor([10, 20, 30, 40])]
    )
    @pytest.mark.parametrize("resample", ['bilinear'])  # TODO: Ignore nearest for now.
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_param(self, degrees, translate, scale, shear, resample, align_corners, same_on_batch, device, dtype):
        _degrees = (
            degrees
            if isinstance(degrees, (int, float, list, tuple))
            else nn.Parameter(degrees.clone().to(device=device, dtype=dtype))
        )
        _translate = (
            translate
            if isinstance(translate, (int, float, list, tuple))
            else nn.Parameter(translate.clone().to(device=device, dtype=dtype))
        )
        _scale = (
            scale
            if isinstance(scale, (int, float, list, tuple))
            else nn.Parameter(scale.clone().to(device=device, dtype=dtype))
        )
        _shear = (
            shear
            if isinstance(shear, (int, float, list, tuple))
            else nn.Parameter(shear.clone().to(device=device, dtype=dtype))
        )

        torch.manual_seed(0)
        input = torch.randint(255, (2, 3, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomAffine(
            _degrees,
            _translate,
            _scale,
            _shear,
            resample,
            align_corners=align_corners,
            same_on_batch=same_on_batch,
            p=1.0,
        )

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=10)
            loss = mse(output, torch.ones_like(output) * 2)
            try:
                loss.backward()
            except Exception as e:
                assert False, (list(aug.named_parameters()), e)
            opt.step()

        if not isinstance(degrees, (int, float, list, tuple)):
            assert isinstance(aug._param_generator.degrees, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug._param_generator.degrees.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug._param_generator.degrees._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (degrees.to(device=device, dtype=dtype) - aug._param_generator.degrees.data).sum() == 0
            else:
                assert (degrees.to(device=device, dtype=dtype) - aug._param_generator.degrees.data).sum() != 0
        if not isinstance(translate, (int, float, list, tuple)):
            assert isinstance(aug._param_generator.translate, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug._param_generator.translate.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug._param_generator.translate._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (translate.to(device=device, dtype=dtype) - aug._param_generator.translate.data).sum() == 0
            else:
                assert (translate.to(device=device, dtype=dtype) - aug._param_generator.translate.data).sum() != 0
        if not isinstance(scale, (int, float, list, tuple)):
            assert isinstance(aug._param_generator.scale, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug._param_generator.scale.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug._param_generator.scale._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (scale.to(device=device, dtype=dtype) - aug._param_generator.scale.data).sum() == 0
            else:
                assert (scale.to(device=device, dtype=dtype) - aug._param_generator.scale.data).sum() != 0
        if not isinstance(shear, (int, float, list, tuple)):
            assert isinstance(aug._param_generator.shear, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug._param_generator.shear.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug._param_generator.shear._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (shear.to(device=device, dtype=dtype) - aug._param_generator.shear.data).sum() == 0
            else:
                assert (shear.to(device=device, dtype=dtype) - aug._param_generator.shear.data).sum() != 0


class TestRandomRotationBackward:
    @pytest.mark.parametrize("degrees", [10, [10.0, 20.0], (10, 20), torch.tensor(10.0), torch.tensor([10, 20])])
    @pytest.mark.parametrize("resample", ['bilinear'])  # TODO: Ignore nearest for now.
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_param(self, degrees, resample, align_corners, same_on_batch, device, dtype):
        _degrees = (
            degrees
            if isinstance(degrees, (int, float, list, tuple))
            else nn.Parameter(degrees.clone().to(device=device, dtype=dtype))
        )

        torch.manual_seed(0)
        input = torch.randint(255, (2, 3, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomRotation(_degrees, resample, align_corners=align_corners, same_on_batch=same_on_batch)

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=10)
            loss = mse(output, torch.ones_like(output) * 2)
            loss.backward()
            opt.step()

        if not isinstance(degrees, (int, float, list, tuple)):
            assert isinstance(aug._param_generator.degrees, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug._param_generator.degrees.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug._param_generator.degrees._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (degrees.to(device=device, dtype=dtype) - aug._param_generator.degrees.data).sum() == 0
            else:
                assert (degrees.to(device=device, dtype=dtype) - aug._param_generator.degrees.data).sum() != 0


class TestRandomPerspectiveBackward:
    @pytest.mark.parametrize("distortion_scale", [0.5, torch.tensor(0.5)])
    @pytest.mark.parametrize("resample", ['bilinear'])  # TODO: Ignore nearest for now.
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_param(self, distortion_scale, resample, align_corners, same_on_batch, device, dtype):
        _distortion_scale = (
            distortion_scale
            if isinstance(distortion_scale, (float, int))
            else nn.Parameter(distortion_scale.clone().to(device=device, dtype=dtype))
        )

        torch.manual_seed(0)
        input = torch.randint(255, (2, 3, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomPerspective(
            _distortion_scale, resample=resample, same_on_batch=same_on_batch, align_corners=align_corners, p=1.0
        )

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=0.1)
            loss = mse(output, torch.ones_like(output) * 2)
            loss.backward()
            opt.step()

        if not isinstance(distortion_scale, (float, int)):
            assert isinstance(aug._param_generator.distortion_scale, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug._param_generator.distortion_scale.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug._param_generator.distortion_scale._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (
                    distortion_scale.to(device=device, dtype=dtype) - aug._param_generator.distortion_scale.data
                ).sum() == 0
            else:
                assert (
                    distortion_scale.to(device=device, dtype=dtype) - aug._param_generator.distortion_scale.data
                ).sum() != 0


class TestRandomMotionBlurBackward:
    @pytest.mark.parametrize("angle", [20.0, torch.tensor([-20.0, 20.0])])
    @pytest.mark.parametrize("direction", [[-0.5, 0.5], torch.tensor([-0.5, 0.5])])
    @pytest.mark.parametrize("border_type", ['constant', 'reflect', 'replicate', 'circular'])
    @pytest.mark.parametrize("resample", ['bilinear'])  # TODO: Ignore nearest for now.
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_param(self, angle, direction, border_type, resample, same_on_batch, device, dtype):
        _angle = (
            angle
            if isinstance(angle, (float, int, list, tuple))
            else nn.Parameter(angle.clone().to(device=device, dtype=dtype))
        )
        _direction = (
            direction
            if isinstance(direction, (list, tuple))
            else nn.Parameter(direction.clone().to(device=device, dtype=dtype))
        )

        torch.manual_seed(0)
        input = torch.randint(255, (2, 3, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomMotionBlur((3, 3), _angle, _direction, border_type, resample, same_on_batch, p=1.0)

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=0.1)
            loss = mse(output, torch.ones_like(output) * 2)
            loss.backward()
            opt.step()

        if not isinstance(angle, (float, int, list, tuple)):
            assert isinstance(aug._param_generator.angle, torch.Tensor)
            if resample == 'nearest' and aug._param_generator.angle.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug._param_generator.angle._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (angle.to(device=device, dtype=dtype) - aug._param_generator.angle.data).sum() == 0
            else:
                # Assert if param not updated
                assert (angle.to(device=device, dtype=dtype) - aug._param_generator.angle.data).sum() != 0
        if not isinstance(direction, (list, tuple)):
            assert isinstance(aug._param_generator.direction, torch.Tensor)
            if torch.all(aug._param_generator.direction._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (direction.to(device=device, dtype=dtype) - aug._param_generator.direction.data).sum() == 0
            else:
                # Assert if param not updated
                assert (direction.to(device=device, dtype=dtype) - aug._param_generator.direction.data).sum() != 0


class TestRandomSharpnessBackward:
    @pytest.mark.parametrize("sharpness", [0.5, [0, 0.5], torch.tensor([0, 0.5])])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_param(self, sharpness, same_on_batch, device, dtype):
        _sharpness = (
            sharpness
            if isinstance(sharpness, (float, int, list, tuple))
            else nn.Parameter(sharpness.clone().to(device=device, dtype=dtype))
        )

        torch.manual_seed(0)
        input = torch.randint(255, (2, 3, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomSharpness(_sharpness, same_on_batch=same_on_batch)

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=0.1)
            loss = mse(output, torch.ones_like(output) * 2)
            loss.backward()
            opt.step()

        if not isinstance(sharpness, (float, int, list, tuple)):
            assert isinstance(aug._param_generator.sharpness, torch.Tensor)
            # Assert if param not updated
            assert (sharpness.to(device=device, dtype=dtype) - aug._param_generator.sharpness.data).sum() != 0


class TestRandomResizedCropBackward:
    @pytest.mark.skip("Param gen is probably breaking grads.")
    @pytest.mark.parametrize("scale", [[0.08, 1.0], torch.tensor([0.08, 1.0])])
    @pytest.mark.parametrize("ratio", [[3.0 / 4.0, 4.0 / 3.0], torch.tensor([3.0 / 4.0, 4.0 / 3.0])])
    @pytest.mark.parametrize("resample", ['bilinear'])  # TODO: Ignore nearest for now.
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_param(self, scale, ratio, resample, align_corners, same_on_batch, device, dtype):
        _scale = (
            scale if isinstance(scale, (list, tuple)) else nn.Parameter(scale.clone().to(device=device, dtype=dtype))
        )
        _ratio = (
            ratio if isinstance(ratio, (list, tuple)) else nn.Parameter(ratio.clone().to(device=device, dtype=dtype))
        )

        torch.manual_seed(0)
        input = torch.randint(255, (2, 3, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomResizedCrop(
            (8, 8), _scale, _ratio, resample=resample, same_on_batch=same_on_batch, align_corners=align_corners
        )

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=0.1)
            loss = mse(output, torch.ones_like(output) * 2)
            loss.backward()
            opt.step()

        if not isinstance(scale, (list, tuple)):
            assert isinstance(aug.scale, torch.Tensor)
            # Assert if param not updated
            assert (scale.to(device=device, dtype=dtype) - aug.scale.data).sum() != 0
        if not isinstance(ratio, (list, tuple)):
            assert isinstance(aug.ratio, torch.Tensor)
            # Assert if param not updated
            assert (ratio.to(device=device, dtype=dtype) - aug.ratio.data).sum() != 0


class TestRandomErasingBackward:
    @pytest.mark.skip("Need differentiable indexing.")
    @pytest.mark.parametrize("scale", [[0.02, 0.33], torch.tensor([0.02, 0.33])])
    @pytest.mark.parametrize("ratio", [[0.3, 3.3], torch.tensor([0.3, 3.3])])
    @pytest.mark.parametrize("value", [0.0])
    @pytest.mark.parametrize("same_on_batch", [True, False])
    def test_param(self, scale, ratio, value, same_on_batch, device, dtype):
        _scale = (
            scale if isinstance(scale, (list, tuple)) else nn.Parameter(scale.clone().to(device=device, dtype=dtype))
        )
        _ratio = (
            ratio if isinstance(ratio, (list, tuple)) else nn.Parameter(ratio.clone().to(device=device, dtype=dtype))
        )

        torch.manual_seed(0)
        input = torch.randint(255, (2, 3, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomErasing(_scale, _ratio, value, same_on_batch)
        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=0.1)
            loss = mse(output, torch.ones_like(output) * 2)
            loss.backward()
            opt.step()

        if not isinstance(scale, (list, tuple)):
            assert isinstance(aug.scale, torch.Tensor)
            if torch.all(aug.scale._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (scale.to(device=device, dtype=dtype) - aug.scale.data).sum() == 0
            else:
                # Assert if param not updated
                assert (scale.to(device=device, dtype=dtype) - aug.scale.data).sum() != 0
        if not isinstance(ratio, (list, tuple)):
            assert isinstance(aug.ratio, torch.Tensor)
            if torch.all(aug.ratio._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (ratio.to(device=device, dtype=dtype) - aug.ratio.data).sum() == 0
            else:
                # Assert if param not updated
                assert (ratio.to(device=device, dtype=dtype) - aug.ratio.data).sum() != 0
