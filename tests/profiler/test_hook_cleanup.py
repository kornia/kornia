# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from torch import Tensor, nn

from kornia.profiler.collectors.layer import LayerCollector


class SimpleModel(nn.Module):
    """Simple fully connected model for hook cleanup tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the fully connected layer."""
        return self.fc(x)


def test_hooks_removed_after_normal_cleanup() -> None:
    """Verify that normal cleanup removes registered forward hooks."""
    model = SimpleModel()
    collector = LayerCollector(layers=["fc"])

    collector.attach(model)
    assert len(model.fc._forward_hooks) > 0, "Hook was not registered"

    collector.remove_hooks()

    assert len(model.fc._forward_hooks) == 0, "Normal cleanup failed"


def test_hooks_removed_after_exception() -> None:
    """Verify that hooks can be removed after an exception occurs."""
    model = SimpleModel()
    collector = LayerCollector(layers=["fc"])

    collector.attach(model)
    try:
        raise RuntimeError("intentional test exception")
    except RuntimeError:
        pass
    finally:
        collector.remove_hooks()

    assert len(model.fc._forward_hooks) == 0, "Exception-path cleanup failed"


def test_no_features_captured_after_cleanup() -> None:
    """Verify that removed hooks no longer capture model features."""
    model = SimpleModel()
    collector = LayerCollector(layers=["fc"])
    x = torch.randn(2, 4)

    collector.attach(model)
    collector.remove_hooks()

    # Running the model after cleanup must not invoke the removed hook.
    model(x)

    assert collector.features == {}, "Features were captured after hooks were removed"


if __name__ == "__main__":
    tests = [
        test_hooks_removed_after_normal_cleanup,
        test_hooks_removed_after_exception,
        test_no_features_captured_after_cleanup,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All hook cleanup checks passed.")
