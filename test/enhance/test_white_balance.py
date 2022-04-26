import numpy as np

import kornia


class TestNormalize:
    def test_balance(self, device, dtype):
        # image H=3, W=3, C=3
        img = np.array([[[21, 104, 155],
                         [22, 105, 156],
                         [22, 105, 156]],
                        [[21, 104, 155],
                         [22, 105, 156],
                         [22, 105, 156]],
                        [[20, 105, 155],
                         [21, 106, 156],
                         [22, 106, 158]]], dtype='uint8')
        img_after = kornia.enhance.white_balance(img)
        assert img.shape == img_after.shape
