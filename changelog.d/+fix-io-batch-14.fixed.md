`load_image` returns a palette PNG's colours under every load type except `UNCHANGED`, which still returns the
palette indices: it used to return the indices as gray for every type. `RGBA8` takes alpha from the `tRNS` chunk, and
the RGB and gray types composite transparent entries over white, as for any RGBA file. A `tRNS` colour key on an
8-bit gray or RGB PNG now makes the key colour transparent in the same way instead of being ignored. Compositing a
semi-transparent pixel over white rounds to the nearest level instead of truncating, which made it up to one level
too dark. `write_image` writes 4-channel (RGBA) PNGs and accepts `.tif` and extensions in any letter case, such as
`.JPG`, all of which it used to reject. `get_sample_images` saves a download byte for byte instead of decoding it and
re-encoding JPEGs at Pillow's default quality 75, so it no longer needs Pillow; a cache filled by an earlier version
keeps its re-encoded files until they are deleted. It creates its cache directory only when it downloads, not for
local paths, and treats a path as a URL only when its scheme is `http` or `https`, so a local file named
`http_local.png` loads.
