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

from __future__ import annotations

import torch

from kornia.metrics import AverageMeter


class TestAverageMeter:
    def test_initial_state(self):
        m = AverageMeter()
        assert m.val == 0
        assert m.avg == 0
        assert m.count == 0

    def test_update_scalar(self):
        m = AverageMeter()
        m.update(0.8, n=1)
        m.update(0.4, n=1)
        assert abs(m.avg - 0.6) < 1e-6

    def test_update_weighted(self):
        m = AverageMeter()
        m.update(1.0, n=3)
        m.update(0.0, n=1)
        assert abs(m.avg - 0.75) < 1e-6

    def test_update_tensor(self):
        m = AverageMeter()
        m.update(torch.tensor(0.9), n=1)
        # avg property converts tensor to float
        assert isinstance(m.avg, float)
        assert abs(m.avg - 0.9) < 1e-6

    def test_reset(self):
        m = AverageMeter()
        m.update(1.0, n=5)
        m.reset()
        assert m.val == 0
        assert m.avg == 0
        assert m.count == 0
