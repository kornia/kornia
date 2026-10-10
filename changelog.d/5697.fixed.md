`MultiResolutionDetector`, the detector of `SIFTFeature`, `KeyNetHardNet` and `KeyNetAffNetHardNet`, no longer reports
spurious keypoints beside its 15 px border strip. It zeroed the strip before non-maxima suppression, so a pixel next
to the strip whose real maximum lay inside it survived the suppression; it now applies the border to the suppression
output, as it already did for the mask. On the Oxford affine images this replaces 3 to 59 of the 2048 detections.
