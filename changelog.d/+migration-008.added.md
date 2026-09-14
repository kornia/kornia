Documented camera projection-core conventions (`PinholeCamera` frames, integer pixel centres, the two
meanings of depth, the `z = 0` policies) and added executable pins for `kornia.geometry.camera`'s
projection core, with the known scale-rule, aliasing, guard and legacy-API limitations tracked in
dedicated issues. (#4294)
