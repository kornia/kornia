`bbox_to_mask3d` now returns the intersection of the three axis ranges for a box that covers or
overhangs a whole output axis, matching `Boxes3D.to_mask`, instead of filling the entire volume
(#4255, #4303). The old implementation `|`-ed the three broadcast axis slabs and tried to recover the
intersection with a three-way `all()` reduction; that recovery breaks the moment any one slab
covers a whole axis, since the union is then all-true on that axis and the reduction can no
longer see the other two slabs' bounds. Computing the intersection directly with `&` needs no
such recovery step. Interior boxes (the case the old reductions happened to recover correctly)
are unaffected and byte-identical.
