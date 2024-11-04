from typing import Any, Dict

from kornia.core import Device, Module, Tensor, concatenate


def batched_forward(
    model: Module, data: Tensor, device: Device, batch_size: int = 128, **kwargs: Dict[str, Any]
) -> Tensor:
    r"""Convenience function, which allows to run the forward in micro-batches.

    When the just model.forward(data) does not fit into device memory, e.g. on laptop GPU.
    In the end, it transfers the output to the device of the input data tensor.
    E.g. running HardNet on 8000x1x32x32 tensor.

    Args:
        model: Any torch model, which outputs a single tensor as an output.
        data: Input data of Bx(Any) shape.
        device: which device should we run on.
        batch_size: "micro-batch" size.
        **kwargs: any other arguments, which accepts model.

    Returns:
        output of the model.

    Example:
        >>> patches = torch.rand(8000, 1, 32, 32)
        >>> sift = kornia.feature.SIFTDescriptor(32)
        >>> desc_batched = batched_forward(sift, patches, torch.device('cpu'), 128)
        >>> desc = sift(patches)
        >>> assert torch.allclose(desc, desc_batched)
    """
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
        out = concatenate(out_list, 0)
        return out.to(data.device)
    return model(data, **kwargs)
