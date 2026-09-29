`RandomMosaic` with a list box input now gives each tile the padding count of the image it was drawn from. A selected
sample used to take its box count from the wrong source images, so it could drop real boxes or return padding rows as
boxes such as `[7, 0, 7, 0]`, from a direct call and from `AugmentationSequential`. Tensor box inputs are unchanged.
(#4715)
