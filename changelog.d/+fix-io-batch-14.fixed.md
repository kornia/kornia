`load_image` returns an 8-bit palette PNG's colours under every load type except `UNCHANGED`, which still returns
the palette indices: it used to return the indices as gray for every type. `RGBA8` takes alpha from the `tRNS`
chunk, and the RGB and gray types composite transparent entries over white, as for any RGBA file; under these types,
a palette that is malformed or too short for its pixels raises `ValueError` naming the file. 1-, 2- and 4-bit
palette PNGs still raise, because kornia-rs cannot decode them. A `tRNS` colour key on an 8-bit gray or RGB PNG now
makes the key colour transparent in the same way instead of being ignored. Compositing a semi-transparent pixel over
white rounds to the nearest level instead of truncating, which made it up to one level too dark. `write_image` writes
4-channel (RGBA) PNGs and accepts `.tif` and extensions in any letter case, such as `.JPG`, all of which it used to
reject; a channel count the format cannot store raises `ShapeError`, still a `ValueError`, naming the supported
counts instead of kornia-rs's "expected 3 channels". `get_sample_images` saves a download byte for byte instead of
decoding it and re-encoding JPEGs at Pillow's default quality 75, so it no longer needs Pillow; a cache filled by an
earlier version keeps its re-encoded files until they are deleted. It creates its cache directory only when it has to
download, not for local paths, and treats a path as a URL only when its scheme is `http` or `https`, so a local file
named `http_local.png` loads.
