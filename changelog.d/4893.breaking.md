`ColorJitter` now skips its brightness step when the factor is the multiplier's neutral `1` instead of `0`.
Before, a batch whose brightness factors were all `0` came back unchanged while the same sample in a mixed
batch came back black, and the default `brightness=0.0` clamped an out-of-range input to `[0, 1]`. Now a
brightness factor of `0` always gives a black image, and the default configuration no longer clamps an
out-of-range input in the brightness step; contrast and saturation steps still clamp when they run (#4785).
