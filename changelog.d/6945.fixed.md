`relative_camera_motion` and `essential_from_Rt` preserve the working precision of camera poses under autocast, avoiding large errors when subtracting translations to recover a small baseline.
