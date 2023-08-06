import keras_core as keras


class Image:
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

        args = (arg.tensor if isinstance(arg, Image) else arg for arg in args)
        kwargs = {key: (value.tensor if isinstance(value, Image) else value) for key, value in kwargs.items()}

        result = func(*args, **kwargs)

        return Image(result, self._image_data_format, self._color_space)

    def __tf_tensor_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = (arg.tensor if isinstance(arg, Image) else arg for arg in args)
        kwargs = {key: (value.tensor if isinstance(value, Image) else value) for key, value in kwargs.items()}

        result = func(*args, **kwargs)

        return Image(result, self._image_data_format, self._color_space)

    def __jax_array_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = (arg.array if isinstance(arg, Image) else arg for arg in args)
        kwargs = {key: (value.array if isinstance(value, Image) else value) for key, value in kwargs.items()}

        result = func(*args, **kwargs)

        return Image(result, self._image_data_format, self._color_space)

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
    def clone(self):
        """Returns a copy of the image."""
        return Image(self._tensor, self._image_data_format, self._color_space)

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
        """Loads an image from a file."""
        raise NotImplementedError

    def to_file(self):
        """Saves an image to a file."""
        raise NotImplementedError

    def to_color_space(self, color_space):
        """Converts the color space of the image."""
        raise NotImplementedError

    def print(self):
        """Prints the contents of the image."""
        print(f"Type: {type(self._tensor)}\nData: {self._tensor}")
