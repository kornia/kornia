`kornia.io.load_image` and `write_image` work on the kornia_rs that a plain `pip install kornia`
resolves. kornia_rs 0.1.11 moved its image readers and writers from the package root into
`kornia_rs.io`, and kornia kept calling the root, so on 0.1.11 and newer `load_image` raised
`AttributeError` for every JPEG and every PNG that is not plain RGB, and `write_image` raised for
TIFF, 16-bit and float32 output (8-bit JPEG and PNG writes still worked); the lock's 0.1.10 kept CI
green. kornia now calls `kornia_rs.io` and requires kornia_rs 0.1.14 (see *Breaking changes*). That
decoder also reads 16-bit PNG/TIFF and float TIFF, which 0.1.10 rejected: `ImageLoadType.UNCHANGED`
returns them as `uint16`/`float32`, and the 8-bit load types raise a `NotImplementedError` that says
so instead of mislabelling the tensor. The nightly PyPI job now imports the installed wheel from
outside the checkout (it used to import the source tree) and round-trips an image instead of only
importing. (#4326)
