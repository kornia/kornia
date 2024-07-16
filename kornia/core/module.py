import torch
import numpy as np
from PIL import Image
from functools import wraps

from .importer import PILImporter, NumpyImporter


class ImageModule(torch.nn.Module):
    """Handles image-based operations.

    This modules accepts multiple input and output data types, provides end-to-end
    visualization, file saving features.
    """
    def __init__(self):
        super(ImageModule, self).__init__()

    def convert_input_output(self, input_names_to_handle=None, output_type='tensor'):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # If input_names_to_handle is None, handle all inputs
                if input_names_to_handle is None:
                    # Convert all args to tensors
                    args = tuple(self.to_tensor(arg) if isinstance(arg, (np.ndarray, Image.Image, torch.Tensor)) else arg for arg in args)
                    # Convert all kwargs to tensors
                    kwargs = {k: self.to_tensor(v) if isinstance(v, (np.ndarray, Image.Image, torch.Tensor)) else v for k, v in kwargs.items()}
                else:
                    # Convert specified args to tensors
                    args = list(args)
                    for i, (arg, name) in enumerate(zip(args, func.__code__.co_varnames)):
                        if name in input_names_to_handle:
                            args[i] = self.to_tensor(arg)
                    # Convert specified kwargs to tensors
                    for name, value in kwargs.items():
                        if name in input_names_to_handle:
                            kwargs[name] = self.to_tensor(value)

                # Call the actual forward method
                tensor_outputs = func(*args, **kwargs)

                if not isinstance(tensor_outputs, tuple):
                    tensor_outputs = (tensor_outputs,)

                # Convert outputs to the desired type
                outputs = []
                for output in tensor_outputs:
                    if output_type == 'tensor':
                        outputs.append(output)
                    elif output_type == 'numpy':
                        outputs.append(self.to_numpy(output))
                    elif output_type == 'pil':
                        outputs.append(self.to_pil(output))
                    else:
                        raise ValueError("Output type not supported. Choose from 'tensor', 'numpy', or 'pil'.")

                return outputs if len(outputs) > 1 else outputs[0]
            return wrapper
        return decorator

    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        elif isinstance(x, Image.Image):
            return torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255  # Convert PIL to tensor
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise TypeError("Input type not supported")

    def to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, Image.Image):
            return np.array(x)
        else:
            raise TypeError("Input type not supported")

    def to_pil(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy() * 255
            if x.dim() == 3:
                x = x.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
                return Image.fromarray(x.astype(np.uint8))
            elif x.dim() == 4:
                x = x.transpose(0, 2, 3, 1)  # Convert from (C, H, W) to (H, W, C)
                return [Image.fromarray(_x.astype(np.uint8)) for _x in x]
            else:
                raise NotImplementedError
        elif isinstance(x, np.ndarray):
            raise NotImplementedError
        elif isinstance(x, Image.Image):
            return x
        else:
            raise TypeError("Input type not supported")

    def _default_visualizer(self, *args, **kwargs, backend="matplotlib"):
        output = self.__call__(*args, **kwargs)
        raise NotImplementedError

    def show(self, *args, **kwargs):
        self._default_visualizer(*args, **kwargs)

    def __call__(self, *inputs, input_names_to_handle=None, output_type='tensor', **kwargs):
        # Wrap the forward method with the decorator
        decorated_forward = self.convert_input_output(
            input_names_to_handle=input_names_to_handle, output_type=output_type
        )(super().__call__)
        return decorated_forward(*inputs, **kwargs)
