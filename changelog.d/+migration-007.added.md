Documented camera distortion and calibration conventions (normalized versus pixel inputs, the
coefficient layout, the `new_K`/`K` roles, the iterative inverses) and added executable pins for
`kornia.geometry.camera`'s distortion models and `kornia.geometry.calibration`, with the
tilt-projection, Kannala-Brandt Jacobian and float16 defects tracked in dedicated issues. (#4312)
