Build sensor-tilt projection matrices directly from their entries, avoiding temporary rotation matrices and matrix multiplications in camera distortion and undistortion.

Direct float32 tilt-projection calls now also retain float32 output under TorchScript autocast, matching eager evaluation.
