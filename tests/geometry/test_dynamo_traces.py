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

from kornia.geometry.plane import Hyperplane
from kornia.geometry.vector import Scalar, Vector2, Vector3

# Safely check for dynamo without relying on internal Kornia paths
pytestmark = pytest.mark.skipif(not hasattr(torch, "compile"), reason="Dynamo (torch.compile) is not available")


def test_eager_backend_traces_geometry():
    t = torch.rand(4)
    plane = Hyperplane(Vector3(torch.tensor([0.0, 0.0, 1.0])), Scalar(torch.tensor(0.5)))

    # Define the 3 cases that were breaking fullgraph compilation
    def fn_vec3(x):
        return Vector3.from_coords(x, x, x).data

    def fn_vec2(x):
        return Vector2.from_coords(x, x).data

    def fn_plane(p):
        return plane.signed_distance(p)

    # Test Vector3
    torch._dynamo.reset()
    compiled_vec3 = torch.compile(fn_vec3, backend="eager", fullgraph=True)
    compiled_vec3(t)  # Should not raise

    # Test Vector2
    torch._dynamo.reset()
    compiled_vec2 = torch.compile(fn_vec2, backend="eager", fullgraph=True)
    compiled_vec2(t)  # Should not raise

    # Test Hyperplane
    torch._dynamo.reset()
    compiled_plane = torch.compile(fn_plane, backend="eager", fullgraph=True)
    compiled_plane(torch.rand(3))  # Should not raise
