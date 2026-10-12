`kornia.image.ColorSpace` is exported, so a `PixelFormat` or a colour space for `Image.from_numpy` no longer needs an
import from `kornia.image.base`. `draw_point2d` and `draw_line` accept a single-value colour, of shape `(1,)` or 0-d,
and write it to every channel, as `draw_rectangle` and `draw_convex_polygon` already did; they used to raise for it on
an image with more than one channel. OpenCV's and Pillow's scalar colours fill only the first channel.
