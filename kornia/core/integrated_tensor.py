import keras_core as keras

@keras.saving.register_keras_serializable("kornia", "integrated_tensor")
class IntegratedTensor:
    def __init__(self, tensor, image_data_format='channels_last', color_space='rgb'):

        self.COLOR_SPACES = ['rgb', 'bgr', 'grayscale', 'hls', 'hsv', 'lab', 'luv', 'xyz', 'ycrcb', 'yuv']

        if image_data_format not in ['channels_last', 'channels_first']:
            raise ValueError(
                "Supplied `image_data_format` is not valid. " "Must be one of `channels_last` or `channels_first`."
            )

        self._image_data_format = image_data_format

        if color_space not in self.COLOR_SPACES:
            raise ValueError("Supplied `color_space` is not valid. " f"Must be one of {self.COLOR_SPACES}.")

        if color_space == 'grayscale':
            if self.image_data_format == 'channels_last' and tensor.shape[-1] != 1:
                raise ValueError("Supplied `image` is not a valid grayscale image. " f"Check the number of channels.")
            elif self.image_data_format == 'channels_first' and tensor.shape[1] != 1:
                raise ValueError("Supplied `image` is not a valid grayscale image. " f"Check the number of channels.")
            
        # TODO: Find better way to handle nested lists with different dtypes present in single tensor
        if isinstance(tensor, list):
            tensor = self.create(tensor)

        self._color_space = color_space
        self._tensor = tensor
        self._backend = keras.backend.backend()

    def __repr__(self):
        return f"Image: {self._tensor.shape}, Data Format: {self._image_data_format}"

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = (arg.tensor if isinstance(arg, IntegratedTensor) else arg for arg in args)
        kwargs = {key: (value.tensor if isinstance(value, IntegratedTensor) else value) for key, value in kwargs.items()}

        result = func(*args, **kwargs)

        return IntegratedTensor(result, self._image_data_format, self._color_space)

    def __tf_tensor_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = (arg.tensor if isinstance(arg, IntegratedTensor) else arg for arg in args)
        kwargs = {key: (value.tensor if isinstance(value, IntegratedTensor) else value) for key, value in kwargs.items()}

        result = func(*args, **kwargs)

        return IntegratedTensor(result, self._image_data_format, self._color_space)

    def __jax_array_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = (arg.array if isinstance(arg, IntegratedTensor) else arg for arg in args)
        kwargs = {key: (value.array if isinstance(value, IntegratedTensor) else value) for key, value in kwargs.items()}

        result = func(*args, **kwargs)

        return IntegratedTensor(result, self._image_data_format, self._color_space)
    
    def __add__(self, other):
        return IntegratedTensor(
            (self._tensor + other.tensor), 
            image_data_format=self._image_data_format, 
            color_space=self._color_space
        )
    
    def __sub__(self, other):
        return IntegratedTensor(
            (self._tensor - other.tensor), 
            image_data_format=self._image_data_format, 
            color_space=self._color_space
        )
    
    def __mul__(self, other):
        return IntegratedTensor(
            (self._tensor * other.tensor), 
            image_data_format=self._image_data_format, 
            color_space=self._color_space
        )

    def __pow__(self, other):
        return IntegratedTensor(
            (self._tensor ** other.tensor), 
            image_data_format=self._image_data_format, 
            color_space=self._color_space
        )

    def __truediv__(self, other):
        return IntegratedTensor(
            (self._tensor / other.tensor), 
            image_data_format=self._image_data_format, 
            color_space=self._color_space
        )
    
    def __getitem__(self, index):
        return self.tensor[index]
    
    def __setitem__(self, index, value):
        backend = keras.backend.backend()

        if backend == "tensorflow":
            from tensorflow.compiler.tf2xla.python import xla
            xla.dynamic_update_slice(self.tensor, update=value, start_indices=index)
        elif backend == "jax":
            from jax import numpy as jnp
            self.tensor = self.tensor.at[index].set(value)
        else:
            self.tensor[index] = value

    @property
    def tensor(self):
        """Returns the tensor of the image."""
        return self._tensor

    @property
    def image_data_format(self):
        """Returns the image data format."""
        return self._image_data_format

    @property
    def color_space(self):
        """Returns the color space of the image."""
        return self._color_space

    @property
    def backend(self):
        """Returns the current backend set for Keras"""
        return self._backend

    @property
    def clone(self, dtype=keras.config.floatx()):
        """Returns a copy of the image with the chosen data type."""
        tensor = self.create(
            self.tensor,
            dtype=dtype
        )
        return IntegratedTensor(tensor, self._image_data_format, self._color_space)

    @property
    def shape(self):
        """Returns the shape of the image."""
        return self._tensor.shape

    @property
    def height(self):
        """Returns the height of the image."""
        if self.image_data_format == 'channels_last':
            return int(self._tensor.shape[1])
        else:
            return int(self._tensor.shape[2])

    @property
    def width(self):
        """Returns the width of the image."""
        if self.image_data_format == 'channels_last':
            return int(self._tensor.shape[2])
        else:
            return int(self._tensor.shape[3])

    @property
    def shape(self):
        """Returns the shape of the image."""
        return self._tensor.shape

    @property
    def dtype(self):
        """Returns the dtype of the image."""
        return self._tensor.dtype

    @property
    def channels(self):
        """Returns the number of channels of the image."""
        if self.image_data_format == 'channels_last':
            return int(self._tensor.shape[3])
        else:
            return int(self._tensor.shape[1])

    @property
    def image_size(self):
        """Returns the image size of the image."""
        if self.image_data_format == 'channels_last':
            return int(self._tensor.shape[1]), int(self._tensor.shape[2])
        else:
            return int(self._tensor.shape[2]), int(self._tensor.shape[3])

    def to_tensor(self):
        """Returns the raw tensor of the image."""
        return self._tensor

    def flip_data_format(self):
        if self._image_data_format == 'channels_last':
            self._tensor = keras.ops.transpose(self._tensor, axes=(0, 3, 1, 2))
            self._image_data_format = 'channels_first'
        else:
            self._tensor = keras.ops.transpose(self._tensor, axes=(0, 2, 3, 1))
            self._image_data_format = 'channels_last'

    def from_file(self, path):
        """TODO: Loads an image from a file."""
        raise NotImplementedError

    def to_color_space(self, color_space):
        """TODO: Converts the color space of the image."""
        raise NotImplementedError
    
    def write(self):
        """TODO: Write the underlying tensor and its configuration to a file"""
        raise NotImplementedError

    def device(self):
        """Get the device name where the underlying tensor is located."""
        print(self.tensor.device)

    def print(self):
        """Prints the contents of the image."""
        print(f"Type: {type(self._tensor)}\nData: {self._tensor}")

    def float(self):
        """Convert the underlying tensor to float"""
        raise NotImplementedError
    
    def get_config(self):
        config_dict = {
            "tensor": self._tensor,
            "image_data_format": self._image_data_format,
            "color_space": self._color_space,
        }
        return config_dict
    
    def create(self, data, dtype='float'):
        backend = keras.backend.backend()

        if backend == "tensorflow":
            from tensorflow import convert_to_tensor
            tensor = convert_to_tensor(data, dtype=dtype)
        elif backend == "numpy":
            import numpy as np
            tensor = np.array(data, dtype=dtype)
        elif backend == "torch":
            from torch import tensor, Tensor
            if isinstance(data, Tensor):
                tensor = data.clone().detach()
            else:
                tensor = tensor(data, dtype=dtype)
        elif backend == "jax":
            from jax.numpy import asarray
            tensor = asarray(data, dtype=dtype)

        return tensor
    
    def erf(self):
        backend = keras.backend.backend()

        if backend == "tensorflow":
            from tensorflow import math
            tensor = math.erf(self._tensor)
        elif backend == "numpy":
            from scipy.special import erf
            tensor = erf(self._tensor)
        elif backend == "torch":
            from torch import erf
            tensor = erf(self._tensor)
        elif backend == "jax":
            from jax.lax import erf
            tensor = erf(self._tensor)

        return tensor
    
    def linalg_solve(self, other):
        backend = keras.backend.backend()
        
        if backend == "tensorflow":
            import tensorflow as tf
            tensor = tf.linalg.solve(self._tensor, other)

        elif backend == "numpy":
            import numpy as np
            tensor = np.linalg.solve(self._tensor, other)
        
        elif backend == "torch":
            import torch
            tensor = torch.linalg.solve(self._tensor, other)

        elif backend == "jax":
            import jax.numpy as jnp
            tensor = jnp.linalg.solve(self._tensor, other)
        return tensor    

    def __len__(self):
        return len(self._tensor)
    
    def __ge__(self, other):
        return self._tensor >= other.tensor
    
    def __gt__(self, other):
        return self._tensor > other.tensor
    
    def __ne__(self, other):
        return self._tensor != other.tensor

    def __eq__(self, other):
        return self._tensor == other.tensor
    
    def __le__(self, other):
        return self._tensor <= other.tensor
    
    def __lt__(self, other):
        return self._tensor < other.tensor
    
    def __sqrt__(self):
        return keras.ops.sqrt(self._tensor)
    
    def T(self):
        return self._tensor.T