`MultiResolutionDetector.detect` apportions `num_features` across pyramid levels with largest-remainder
(Hamilton) apportionment instead of flooring each level's fractional share independently (#4101). The floors
used to discard every fractional part, so a small `num_features` could round the whole apportionment down: with
the default configuration the six shares are `0.508 .. 0.016`, and every level but the finest lost its entire
quota to truncation, so the detector effectively searched a single scale. The shortfall -- `num_features` minus
the sum of the floors -- is now handed to the levels with the largest fractional remainder, one slot each, so
the quotas always sum to exactly `num_features` and stay spread across scales. #4098's narrower `num_features=1`
fallback, which handed the request's one slot to the largest-share level when every quota floored to zero, is
the special case where the shortfall equals `num_features` and is superseded by the general apportionment.
The apportionment was then removed again by #4222, inside this same unreleased window, so no release
ever carries it; the `num_features=1` behaviour it fixed still holds.
