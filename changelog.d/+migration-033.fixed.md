`elastic_transform2d` now builds its identity sampling grid with the requested `align_corners`
convention, so a zero displacement field preserves the input image. (closes #4235). (#4382)
