`rotate3d` builds its default center in the angles' floating-point dtype, so mixed-precision autocast calls match the equivalent explicit-center call.
