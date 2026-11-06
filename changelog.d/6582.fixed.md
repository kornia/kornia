`Image.to_numpy()` now exports bfloat16 data as a float32 array, preserving the image's stored layout and values instead of raising an unsupported-dtype error.
