from itertools import product

import pytest


@pytest.fixture
def shape(B, C, H, W):
    return (B, C, H, W)


def pytest_generate_tests(metafunc):
    B = [1, 5]
    C = [1, 3]
    H = W = [128]  # , 237, 512]
    params = list(product(B, C, H, W))
    metafunc.parametrize("B, C, H, W", params)
