from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from kornia.augmentation import RandomAffine3D, RandomMotionBlur3D, RandomPerspective3D, RandomRotation3D


class TestRandomAffine3DBackward:
    @pytest.mark.parametrize(
        "degrees",
        [
            10,
            [10.0, 20.0],
            [10.0, 20.0, 30.0],
            [(10, 20), (10, 20), (10, 20)],
            torch.tensor(10.0),
            torch.tensor([10.0, 20.0]),
            torch.tensor([10, 20, 30]),
            torch.tensor([(10, 20), (10, 20), (10, 20)]),
        ],
    )
    @pytest.mark.parametrize("translate", [[0.1, 0.2, 0.3], torch.tensor([0.1, 0.2, 0.3])])
    @pytest.mark.parametrize(
        "scale",
        [
            [0.1, 0.2],
            [(0.1, 0.2), (0.1, 0.2), (0.1, 0.2)],
            torch.tensor([0.1, 0.2]),
            torch.tensor([(0.1, 0.2), (0.1, 0.2), (0.1, 0.2)]),
        ],
    )
    @pytest.mark.parametrize(
        "shear",
        [
            10.0,
            [10.0, 20.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
            torch.tensor(10),
            torch.tensor([10, 20]),
            torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            torch.tensor([(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]),
        ],
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
        input = torch.randint(255, (2, 3, 10, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomAffine3D(
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
            loss = mse(output, torch.ones_like(output) * 2)  # to ensure that a big loss value could be obtained
            loss.backward()
            opt.step()

        if not isinstance(degrees, (int, float, list, tuple)):
            assert isinstance(aug.degrees, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug.degrees.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug.degrees._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (degrees.to(device=device, dtype=dtype) - aug.degrees.data).sum() == 0
            else:
                assert (degrees.to(device=device, dtype=dtype) - aug.degrees.data).sum() != 0
        if not isinstance(translate, (int, float, list, tuple)):
            assert isinstance(aug.translate, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug.translate.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug.translate._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (translate.to(device=device, dtype=dtype) - aug.translate.data).sum() == 0
            else:
                assert (translate.to(device=device, dtype=dtype) - aug.translate.data).sum() != 0
        if not isinstance(scale, (int, float, list, tuple)):
            assert isinstance(aug.scale, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug.scale.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug.scale._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (scale.to(device=device, dtype=dtype) - aug.scale.data).sum() == 0
            else:
                assert (scale.to(device=device, dtype=dtype) - aug.scale.data).sum() != 0
        if not isinstance(shear, (int, float, list, tuple)):
            assert isinstance(aug.shears, torch.Tensor)
            # Assert if param not updated
            if resample == 'nearest' and aug.shears.is_cuda:
                # grid_sample in nearest mode and cuda device returns nan than 0
                pass
            elif resample == 'nearest' or torch.all(aug.shears._grad == 0.0):
                # grid_sample will return grad = 0 for resample nearest
                # https://discuss.pytorch.org/t/autograd-issue-with-f-grid-sample/76894
                assert (shear.to(device=device, dtype=dtype) - aug.shears.data).sum() == 0
            else:
                assert (shear.to(device=device, dtype=dtype) - aug.shears.data).sum() != 0


class TestRandomRotation3DBackward:
    @pytest.mark.parametrize(
        "degrees",
        [
            10,
            [10.0, 20.0],
            [10.0, 20.0, 30.0],
            [(10, 20), (10, 20), (10, 20)],
            torch.tensor(10.0),
            torch.tensor([10.0, 20.0]),
            torch.tensor([10, 20, 30]),
            torch.tensor([(10, 20), (10, 20), (10, 20)]),
        ],
    )
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
        input = torch.randint(255, (2, 3, 10, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomRotation3D(_degrees, resample, align_corners=align_corners, same_on_batch=same_on_batch, p=1.0)

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=10)
            loss = mse(output, torch.ones_like(output) * 2)  # to ensure that a big loss value could be obtained
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


class TestRandomPerspective3DBackward:
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
        input = torch.randint(255, (2, 3, 10, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomPerspective3D(
            _distortion_scale, resample=resample, same_on_batch=same_on_batch, align_corners=align_corners, p=1.0
        )

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=10)
            loss = mse(output, torch.ones_like(output) * 2)  # to ensure that a big loss value could be obtained
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


class TestRandomMotionBlur3DBackward:
    @pytest.mark.parametrize("angle", [20.0, torch.tensor(20.0), torch.tensor([20.0])])
    @pytest.mark.parametrize("direction", [[-0.5, 0.5], torch.tensor([-0.5, 0.5])])
    # 'reflect' is not implemented by torch.
    @pytest.mark.parametrize("border_type", ['constant', 'replicate', 'circular'])
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
        input = torch.randint(255, (2, 3, 10, 10, 10), device=device, dtype=dtype) / 255.0
        aug = RandomMotionBlur3D((3, 3), _angle, _direction, border_type, resample, same_on_batch, p=1.0)

        output = aug(input)

        if len(list(aug.parameters())) != 0:
            mse = nn.MSELoss()
            opt = torch.optim.SGD(aug.parameters(), lr=10)
            loss = mse(output, torch.ones_like(output) * 2)  # to ensure that a big loss value could be obtained
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
