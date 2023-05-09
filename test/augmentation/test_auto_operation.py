import inspect
from typing import List

import pytest
import torch

from kornia.augmentation.auto.autoaugment import AutoAugment
from kornia.augmentation.auto.operations import OperationBase, ops
from kornia.augmentation.auto.rand_augment.rand_augment import RandAugment
from kornia.augmentation.auto.rand_augment.rand_augment import default_policy as randaug_config
from kornia.augmentation.auto.trivial_augment import TrivialAugment
from kornia.augmentation.container import AugmentationSequential
from kornia.geometry.bbox import bbox_to_mask
from kornia.testing import assert_close
from test.augmentation.test_container import reproducibility_test


def _find_all_ops() -> List[OperationBase]:
    _ops = [op for _, op in inspect.getmembers(ops, inspect.isclass)]
    return [op() for op in _ops if issubclass(op, OperationBase) and op != OperationBase]


def _test_sequential(augment_method, device, dtype):
    inp = torch.rand(1, 3, 1000, 500, device=device, dtype=dtype)
    bbox = torch.tensor([[[355, 10], [660, 10], [660, 250], [355, 250]]], device=device, dtype=dtype)
    keypoints = torch.tensor([[[465, 115], [545, 116]]], device=device, dtype=dtype)
    mask = bbox_to_mask(
        torch.tensor([[[155, 0], [900, 0], [900, 400], [155, 400]]], device=device, dtype=dtype), 1000, 500
    )[:, None]
    aug = AugmentationSequential(augment_method, data_keys=["input", "mask", "bbox", "keypoints"])
    out = aug(inp, mask, bbox, keypoints)
    assert out[0].shape == inp.shape
    assert out[1].shape == mask.shape
    assert out[2].shape == bbox.shape
    assert out[3].shape == keypoints.shape
    assert set(out[1].unique().tolist()).issubset(set(mask.unique().tolist()))

    out_inv = aug.inverse(*out)
    assert out_inv[0].shape == inp.shape
    assert out_inv[1].shape == mask.shape
    assert out_inv[2].shape == bbox.shape
    assert out_inv[3].shape == keypoints.shape
    assert set(out_inv[1].unique().tolist()).issubset(set(mask.unique().tolist()))

    reproducibility_test((inp, mask, bbox, keypoints), aug)


class TestOperations:
    @pytest.mark.parametrize("op", _find_all_ops())
    def test_step_routine(self, op: OperationBase):
        op = op.eval()
        op = op.train()
        if op.magnitude is not None:
            init_mag = op.magnitude.item()
        init_prob = op.probability.item()

        optimizer = torch.optim.SGD(op.parameters(), lr=10, momentum=0.9)

        for _ in range(5):
            in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
            optimizer.zero_grad()
            x = op(in_tensor)
            loss = (x - 0).mean()
            loss.backward()
            optimizer.step()

        if op.magnitude is not None:
            assert init_mag != op.magnitude.item()
        if isinstance(op, ops.Equalize):
            # NOTE: Equalize is somehow not working yet to update the probabilities.
            return
        assert init_prob != op.probability.item()


class TestAutoAugment:
    @pytest.mark.parametrize("policy", ["imagenet", "cifar10", "svhn", [[("shear_x", 0.9, 4), ("invert", 0.2, None)]]])
    def test_smoke(self, policy):
        aug = AutoAugment(policy)
        in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
        aug(in_tensor)
        aug.is_intensity_only()

    def test_transform_mat(self):
        aug = AutoAugment([[("shear_x", 0.9, 4), ("invert", 0.2, None)]], transformation_matrix_mode="silence")
        in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
        aug(in_tensor)
        trans = aug.get_transformation_matrix(in_tensor, params=aug._params)
        assert_close(trans, aug.transform_matrix)

    def test_reproduce(self):
        aug = AutoAugment()
        in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
        out_tensor = aug(in_tensor)
        out_tensor_2 = aug(in_tensor, params=aug._params)
        assert_close(out_tensor, out_tensor_2)

    def test_sequential(augment_method, device, dtype):
        _test_sequential(AutoAugment(), device=device, dtype=dtype)


class TestRandAugment:
    @pytest.mark.parametrize("policy", [None, [[("translate_y", -0.5, 0.5)]]])
    def test_smoke(self, policy):
        if policy is None:
            n = len(randaug_config)
        else:
            n = 1
        aug = RandAugment(n=n, m=15, policy=policy)
        in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
        aug(in_tensor)

    def test_transform_mat(self):
        aug = RandAugment(n=3, m=15)
        in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
        aug(in_tensor)
        trans = aug.get_transformation_matrix(in_tensor, params=aug._params)
        assert_close(trans, aug.transform_matrix)

    def test_reproduce(self):
        aug = RandAugment(n=3, m=15)
        in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
        out_tensor = aug(in_tensor)
        out_tensor_2 = aug(in_tensor, params=aug._params)
        assert_close(out_tensor, out_tensor_2)

    def test_sequential(augment_method, device, dtype):
        _test_sequential(RandAugment(n=3, m=15), device=device, dtype=dtype)


class TestTrivialAugment:
    @pytest.mark.parametrize("policy", [None, [[("translate_y", -0.5, 0.5)]]])
    def test_smoke(self, policy):
        aug = TrivialAugment(policy=policy)
        in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
        aug(in_tensor)

    def test_transform_mat(self):
        aug = TrivialAugment()
        in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
        aug(in_tensor)
        aug(in_tensor)
        trans = aug.get_transformation_matrix(in_tensor, params=aug._params)
        assert_close(trans, aug.transform_matrix)

    def test_reproduce(self):
        aug = TrivialAugment()
        in_tensor = torch.rand(10, 3, 50, 50, requires_grad=True)
        out_tensor = aug(in_tensor)
        out_tensor_2 = aug(in_tensor, params=aug._params)
        assert_close(out_tensor, out_tensor_2)

    def test_sequential(augment_method, device, dtype):
        _test_sequential(TrivialAugment(), device=device, dtype=dtype)
