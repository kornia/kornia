`depth_to_normals` now raises `ShapeError` when `H < 2` or `W < 2`, since surface normals need two
tangent directions. Previously, singleton axes could yield zero or non-finite normals, and empty
spatial dimensions failed inside padding. Inputs with both dimensions at least 2 are unchanged. (#4458)
