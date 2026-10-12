`draw_convex_polygon` overwrites NaN and infinite pixels inside the polygon, which used to stay NaN, and fills a
`uint8` image with a float colour, which used to raise `RuntimeError`; the colour is cast to the image dtype. Its
edges are computed in at least float32 and without rounding the slope first, so `bfloat16` or `float16` vertices fill
the right columns (`bfloat16` vertices at x = 260 and 280 used to fill columns 259 to 281), a pixel centre lying
exactly on a sloped edge is filled, and a long, shallow edge whose vertices lie outside the image is placed correctly.
`tensor_to_image` returns a `bfloat16` tensor as a `float32` array instead of raising `TypeError`. `Image.from_numpy`
accepts a 2-D gray array, which raised `IndexError`. `Image.to` and `Image.float` set `bit_depth` from the new dtype,
so `clone()` of a converted image no longer raises `BaseError`. `Image.write` and `Image.print` convert a `BGR` image
to RGB first, so the file and the terminal show the true colours instead of red and blue swapped. `print_image`
accepts any `os.PathLike` path, such as a `pathlib.Path`. Docstrings: the `Image.from_dlpack` example passes a `(C, H,
W)` array, `Image.height` and `Image.width` name rows and columns the right way round, `Image.write` no longer says it
writes only JPEG, and the `image_to_tensor`, `tensor_to_image` and `draw_convex_polygon` argument descriptions are
corrected.
