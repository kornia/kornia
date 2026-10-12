`pixel2cam` now validates the full `Bx4x4` shape of `intrinsics_inv`, rejecting invalid matrix sizes
before they cause unrelated transformation errors or return the wrong number of coordinate components. (#4381)
