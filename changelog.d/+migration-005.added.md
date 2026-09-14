Documented the `kornia.sensors.camera` conventions (`Vector`-typed inputs, the per-model `params`
layout, the shared half-pixel `scale` rule, and the pinhole mapping shared with `kornia.geometry.camera`,
including its camera-axis broadcasting, projection rounding and zero/near-zero depth differences) and
added executable pins, with the unimplemented models tracked in a dedicated issue; the three non-pinhole
models now appear on the sensors documentation page. (#4318)
