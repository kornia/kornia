`RandomRain` now rejects drop heights equal to the image height and absolute drop widths equal to the image
width with the documented validation error, instead of allowing boundary-sized drops to reach an internal
`IndexError`. (#4451)
