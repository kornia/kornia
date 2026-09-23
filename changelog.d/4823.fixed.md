`RandomResizedCrop` now follows torchvision's `get_params` in two places, so `scale=(1.0, 1.0)` keeps the whole
image for square and portrait inputs (an 8x6 input gave a 4x6 crop). A candidate may now equal the input size
instead of having to be strictly smaller, and when none of the ten candidates fits, the fallback size compares the
input's width/height with both `min(ratio)` and `max(ratio)`: an input already in range is kept whole, a narrower
one keeps its full width and a wider one its full height. The fallback previously compared height/width with
`min(ratio)` only, which could return a crop outside both `scale` and `ratio`. Sampled crops change for
configurations that reach the fallback (a `scale` close to `(1.0, 1.0)`), and for draws where a candidate exactly
matches the input width or height, which used to be rejected: about 0.5-1% of draws with the default `scale` and
`ratio` on 224x224 or 256x192 inputs, more on small images. The fallback crop is still placed at a random position,
where torchvision centres it. Fixes #4814.
