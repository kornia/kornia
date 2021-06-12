import torch

import kornia


def batched_forward(
    model: torch.nn.Module, data: torch.Tensor, device: torch.device, batch_size: int = 128, **kwargs
) -> torch.Tensor:
    r'''Convenience function, which allows to run the forward in micro-batches,
    when the just model.forward(data) does not fit into device memory, e.g. on laptop GPU.
    In the end, it transfers the output to the device of the input data tensor.
    E.g. running HardNet on 8000x1x32x32 tensor.
    Args:
        model (torch.nn.Module): Any torch model, which outputs a single tensor as an output
        data (torch.Tensor): Input data of Bx(Any) shape
        device (torch.device): which device should we run on
        batch_size (int): "micro-batch" size. Default: 128
        **kwargs: any other arguments, which accepts model
    Returns:
        torch.Tensor: output of the model

    Example:
        >>> patches = torch.rand(8000, 1, 32, 32)
        >>> sift = kornia.feature.SIFTDescriptor(32)
        >>> desc_batched = batched_forward(sift, patches, torch.device('cpu'), 128)
        >>> desc = sift(patches)
        >>> assert torch.allclose(desc, desc_batched)'''
    model_dev = model.to(device)
    B: int = len(data)
    bs: int = batch_size
    if B > batch_size:
        out_list = []
        n_batches = int(B // bs + 1)
        for batch_idx in range(n_batches):
            st = batch_idx * bs
            if batch_idx == n_batches - 1:
                if (batch_idx + 1) * bs > B:
                    end = B
                else:
                    end = (batch_idx + 1) * bs
            else:
                end = (batch_idx + 1) * bs
            if st >= end:
                continue
            out_list.append(model_dev(data[st:end].to(device), **kwargs))
        out = torch.cat(out_list, dim=0)
        return out.to(data.device)
    return model(data, **kwargs)
