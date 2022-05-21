from time import time

import pytest
import torch

import kornia

shapes = [(512, 3, 256, 256), (256, 1, 64, 64)]
PSs = [224, 32]


@pytest.mark.xfail(reason='May cause memory issues.')
def test_performance_speed(device, dtype):
    if device.type != 'cuda' or not torch.cuda.is_available():
        pytest.skip("Cuda not available in system,")

    print("Benchmarking warp_affine")
    for input_shape in shapes:
        for PS in PSs:
            BS = input_shape[0]
            inpt = torch.rand(input_shape).to(device)
            As = torch.eye(3).unsqueeze(0).repeat(BS, 1, 1)[:, :2, :].to(device)
            As += 0.1 * torch.rand(As.size()).to(device)
            torch.cuda.synchronize(device)
            t = time()
            _ = kornia.warp_affine(inpt, As, (PS, PS))
            print(f"inp={input_shape}, PS={PS}, dev={device}, {time() - t}, sec")
            torch.cuda.synchronize(device)
