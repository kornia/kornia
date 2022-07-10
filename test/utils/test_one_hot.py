import torch

import kornia
from kornia.testing import assert_close


class TestOneHot:
    def test_smoke(self):
        num_classes = 4
        labels = torch.zeros(2, 2, 1, dtype=torch.int64)
        labels[0, 0, 0] = 0
        labels[0, 1, 0] = 1
        labels[1, 0, 0] = 2
        labels[1, 1, 0] = 3

        # convert labels to one hot tensor
        one_hot = kornia.utils.one_hot(labels, num_classes)

        assert_close(one_hot[0, labels[0, 0, 0], 0, 0], 1.0)
        assert_close(one_hot[0, labels[0, 1, 0], 1, 0], 1.0)
        assert_close(one_hot[1, labels[1, 0, 0], 0, 0], 1.0)
        assert_close(one_hot[1, labels[1, 1, 0], 1, 0], 1.0)
