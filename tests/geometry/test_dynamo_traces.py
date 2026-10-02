import pytest
import torch

from kornia.geometry.plane import Hyperplane
from kornia.geometry.vector import Scalar, Vector2, Vector3

# Safely check for dynamo without relying on internal Kornia paths
pytestmark = pytest.mark.skipif(
    not hasattr(torch, "compile"), reason="Dynamo (torch.compile) is not available"
)

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