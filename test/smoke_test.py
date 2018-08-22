import unittest

import torch
import torchgeometry


class Tester(unittest.TestCase):

    def test_smoke(self):
        x = torch.rand(1, 2, 3)
        assert x.shape == (1, 2, 3), x.shape


if __name__ == '__main__':
    unittest.main()
