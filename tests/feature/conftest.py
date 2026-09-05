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

import pytest
import torch


@pytest.fixture()
def cudnn_tf32_follows_option(request):
    """Compute convolutions in real float32 on CUDA, so a float32 tolerance means float32.

    The root ``conftest.py``'s ``--tf32`` option drives only ``set_float32_matmul_precision``, which
    does not reach cuDNN: ``torch.backends.cudnn.allow_tf32`` keeps PyTorch's ``True`` default in
    every run. TF32 rounds a convolution's inputs to 10 mantissa bits, and eager and inductor do not
    have to pick the same kernel, so on a deep conv stack the two disagree by ~2e-4 -- twenty times
    ``assert_close``'s float32 ``atol`` -- while the same comparison agrees to 6e-7 once cuDNN is in
    real float32. A compile-vs-eager test that does not take TF32 out of the picture is measuring the
    backend rather than the graph.

    The flag follows ``--tf32`` rather than being forced off, so a ``--tf32`` run still gets TF32 in
    cuDNN as well as in matmul. The repo-wide fix is for ``pytest_sessionstart`` to set
    ``cudnn.allow_tf32`` from the same option; ``tests/color/test_yuv.py`` carries the same local
    workaround for the same reason.

    The alternative -- keeping TF32 on and widening the CUDA float32 bound to ~1e-3, TF32's own
    mantissa -- was tried and dropped: it leaves the assertion a hundred times looser than the signal
    it exists to check, so a genuine compile-vs-eager divergence below 1e-3 would pass. The objection
    to this route is that inductor's on-disk FX graph cache is not keyed on ``allow_tf32``, so a
    context entered after a TF32-on compile in the same process can raise out of
    ``assert_tensor_metadata``. That applies to ``torch.backends.cudnn.flags(...)``, which swaps
    several flags at once; assigning the single attribute, as here and in ``test_yuv.py``, does not
    reproduce it -- ``tests/feature -k "dynamo or compile"`` on CUDA float32 is green with this fixture
    on both a cold and a warm inductor cache.
    """
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = bool(request.config.getoption("--tf32"))
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous
