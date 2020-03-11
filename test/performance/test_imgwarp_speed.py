import torch
import kornia as kornia
from time import time


devices = [torch.device('cpu'), torch.device('cuda:0')]
shapes = [(512, 3, 256, 256), (256, 1, 64, 64)]
PSs = [224, 32]

if __name__ == '__main__':
    print("Benchmarking warp_affine")
    for device in devices:
        for input_shape in shapes:
            for PS in PSs:
                BS = input_shape[0]
                inpt = torch.rand(input_shape).to(device)
                As = torch.eye(3).unsqueeze(0).repeat(BS, 1, 1)[:, :2, :].to(device)
                As += 0.1 * torch.rand(As.size()).to(device)
                torch.cuda.synchronize()
                t = time()
                kornia_wa = kornia.warp_affine(inpt, As, (PS, PS))
                print(f"inp={input_shape}, PS={PS}, dev={device}, {time() - t}, sec")
                torch.cuda.synchronize()
