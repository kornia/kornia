import keras_core as keras  # type: ignore

split = keras.ops.split


def channels_axis() -> int:
    image_data_format = keras.backend.image_data_format()
    if image_data_format == 'channels_last':
        return -1
    elif image_data_format == 'channels_first':
        return -3
    else:
        raise NotImplementedError('image data format is not channels last or first.')
