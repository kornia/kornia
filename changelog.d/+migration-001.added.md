Optional-dependency extra `kornia[image]` installs Pillow for PIL-backed image input/output,
display helpers and `kornia.io.sample`; missing-Pillow errors now name the extra to install, and the unused
`DinoVisionTransformer.forward_features_list` list path has been removed. (#4464)
