`image_histogram2d` now raises `TypeError` for a `None` image, like it already does for every other non-tensor input, instead of the incidental `AttributeError` from the image-rank check.
