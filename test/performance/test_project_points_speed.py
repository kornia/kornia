import pytest
from time import time

import torch
import kornia as kornia


points_shapes = [(64, 1024**2, 3), (8192, 8192, 3), (1024**2, 64, 3)]


def test_performance_speed(device, dtype):
    if device.type != 'cuda' or not torch.cuda.is_available():
        pytest.skip("Cuda not available in system,")

    print("Benchmarking project_points")
    for input_shape in points_shapes:
        BS = input_shape[0]
        inpt = torch.rand(input_shape).to(device)
        pose = torch.rand((1, 4, 4)).to(device)
        torch.cuda.synchronize(device)
        t = time()
        kornia.geometry.transform_points(pose, inpt)
        torch.cuda.synchronize(device)
        print(f"inp={input_shape}, dev={device}, {time() - t}, sec")
