import inspect
from typing import List

import pytest
import torch

import kornia.augmentation.auto.operations.ops as ops
from kornia.augmentation.auto.autoaugment import AutoAugment
from kornia.augmentation.auto.operations import OperationBase
from kornia.augmentation.auto.rand_augment.rand_augment import RandAugment
from kornia.augmentation.auto.rand_augment.rand_augment import default_policy as randaug_config
from kornia.augmentation.auto.trivial_augment import TrivialAugment


def _find_all_ops() -> List[OperationBase]:
    _ops = [op for _, op in inspect.getmembers(ops, inspect.isclass)]
    return list([op() for op in _ops if issubclass(op, OperationBase) and op != OperationBase])


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
            in_tensor = torch.rand(10, 3, 10, 10, requires_grad=True)
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
        in_tensor = torch.rand(10, 3, 10, 10, requires_grad=True)
        aug(in_tensor)


class TestRandAugment:
    @pytest.mark.parametrize("policy", [None, [[("translate_y", -0.5, 0.5)]]])
    def test_smoke(self, policy):
        if policy is None:
            n = len(randaug_config)
        else:
            n = 1
        aug = RandAugment(n=n, m=15, policy=policy)
        in_tensor = torch.rand(10, 3, 10, 10, requires_grad=True)
        aug(in_tensor)


class TestTrivialAugment:
    @pytest.mark.parametrize("policy", [None, [[("translate_y", -0.5, 0.5)]]])
    def test_smoke(self, policy):
        aug = TrivialAugment(policy=policy)
        in_tensor = torch.rand(10, 3, 10, 10, requires_grad=True)
        aug(in_tensor)
