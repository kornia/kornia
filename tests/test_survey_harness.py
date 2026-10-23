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

"""Numerical comparisons shared by the export and inductor conformance probes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from testing.conformance import import_survey_module

pytestmark = pytest.mark.device_agnostic


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128], ids=["single", "double"])
@pytest.mark.parametrize("as_tensor", [False, True], ids=["numpy", "torch"])
@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        (1 + 2j, 1 + 2j, True),
        (1 + 2j, 1 + 2.000001j, True),
        (1 + 2j, 1 - 20j, False),
        (1 + 2j, 10 + 2j, False),
        (complex(np.nan, 2), complex(np.nan, 2), True),
        (complex(1, np.nan), complex(1, np.nan), True),
        (complex(np.nan, 2), complex(np.nan, 20), False),
        (complex(1, np.nan), complex(10, np.nan), False),
        (complex(np.nan, 2), complex(1, np.nan), False),
        (complex(1, np.nan), complex(1, 2), False),
        (complex(np.inf, 2), complex(np.inf, 2), True),
        (complex(np.inf, 2), complex(np.inf, 20), False),
    ],
    ids=[
        "equal",
        "within-tolerance",
        "imaginary-mismatch",
        "real-mismatch",
        "real-nan-equal",
        "imaginary-nan-equal",
        "real-nan-imaginary-mismatch",
        "imaginary-nan-real-mismatch",
        "nan-components-differ",
        "nan-pattern-differs",
        "infinity-equal",
        "infinity-imaginary-mismatch",
    ],
)
def test_close_complex(reference, candidate, expected, dtype, as_tensor):
    a, b = np.array([reference], dtype=dtype), np.array([candidate], dtype=dtype)
    if as_tensor:
        a, b = torch.from_numpy(a), torch.from_numpy(b)
    ok, reason = import_survey_module("harness").close(a, b, atol=1e-5, rtol=1e-5)
    assert ok is expected, reason
    assert bool(reason) is not expected
