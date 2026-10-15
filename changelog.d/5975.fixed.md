`kornia.io.write_image` rejects an image whose height or width is zero with a `ShapeError` naming the shape,
before the file is created. It used to reach kornia-rs, which raised a bare `Exception` and left a truncated file
behind (#5770).
