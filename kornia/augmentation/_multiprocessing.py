import torch


class MultiprocessWrapper:
    """Utility class which when used as a base class, makes the class work with the 'spawn' multiprocessing
    context."""

    def __init__(self, *args, **kwargs) -> None:
        args = (arg.clone() if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {key: val.clone() if isinstance(val, torch.Tensor) else val for key, val in kwargs.items()}

        super().__init__(*args, **kwargs)
