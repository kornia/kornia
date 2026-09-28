`kornia_rs>=0.1.14` is required; the floor used to be 0.1.9. kornia_rs 0.1.11 relocated its image I/O
and 0.1.12/0.1.13 lack the libjpeg-turbo reader, so no single call site works across 0.1.9-0.1.14;
rather than dispatch per version, kornia calls the current layout. `pip install -U kornia_rs` on an
environment that pins an older one. (#4326)
