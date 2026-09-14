Make `load_pointcloud_ply` and `load_pointcloud_ply_binary` parse the PLY header instead of skipping a fixed
number of lines. Both loaders skipped `header_size=8` lines and read everything after them as `x y z` triples, so a
minimal PLY file (seven header lines, no `comment`) lost its first point to a header line -- the binary reader
returned header bytes decoded as doubles -- and any file with extra vertex properties (normals, colours) or a
`face` element after the vertices was rejected or misread. The readers now stop at `end_header`, take the point
count from `element vertex N`, select `x`, `y`, `z` by property name whatever their scalar type, skip trailing
elements, and read big-endian payloads. `header_size` is deprecated and ignored (a `DeprecationWarning` is
emitted when it is passed). Files without a proper PLY header, which the old line-skipping loader could read
by accident, are now rejected with a `ValueError`, and so is a file that declares a `list` property among the
vertex properties -- a list has no statically known length, so the columns after it cannot be located. The
binary reader additionally rejects a `list` property in an element that *precedes* the vertices, whose byte
length it would have to know to reach the vertex data; the ASCII reader reads such a file, since every ASCII
element instance is one line whatever it contains.
