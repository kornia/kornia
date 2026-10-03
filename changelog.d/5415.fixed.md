:class:`~kornia.geometry.camera.pinhole.PinholeCamera` ``project`` and ``unproject`` now build their projection
matrix from the intrinsics' top-left 3x3 block, so a 3x3 ``K`` zero-padded to 4x4 round-trips exactly like its
homogeneous embedding instead of projecting while ``unproject`` raises on the singular 4x4 product
(`#4771 <https://github.com/kornia/kornia/issues/4771>`_).
