LoFTR now removes coarse matches near each sample's valid-region border when padding masks are supplied,
instead of measuring that border from the padded canvas. Either image mask can be supplied independently.
Padding is assumed to lie on the right and bottom of a top-left-aligned valid rectangle.
