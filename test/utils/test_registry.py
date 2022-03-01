import pytest
import torch.nn as nn

from kornia.utils.registry import Registry


class TestRegistry:

    def test_register(self):
        reg = Registry("random_reg")
        reg._register_module(Registry)
        assert len(reg) == 1
        assert Registry.__name__ in reg

        with pytest.raises(RuntimeError):
            reg._register_module(Registry)

        reg._register_module(Registry, force=True)

    def test_register_modules_from_namespace(self):
        reg = Registry("random_reg")
        reg.register_modules_from_namespace("kornia.color", allowed_classes=[nn.Module])
        assert len(reg.list_modules()) != 0
        assert len(reg) != 0
