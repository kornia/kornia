from typing import List
import pytest
import inspect

import torch
from kornia.augmentation.meta.operations import OperationBase
import kornia.augmentation.meta.operations.ops as ops


def _find_all_ops() -> List[OperationBase]:
    _ops = [op for _, op in inspect.getmembers(ops, inspect.isclass)]
    return list([op() for op in _ops if issubclass(op, OperationBase) and op != OperationBase])


class TestOperations:

    @pytest.mark.parametrize("op", _find_all_ops())
    def test_step_routine(self, op: OperationBase):
        op = op.train()
        if op.magnitude is not None:
            init_mag = op.magnitude.item()
        init_prob = op.probability.item()

        optimizer = torch.optim.SGD(op.parameters(), lr=10, momentum=0.9)

        for _ in range(2):
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