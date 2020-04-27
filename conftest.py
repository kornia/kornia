import pytest


def pytest_addoption(parser):
    parser.addoption('--typetest', action="store", default="cpu")
    parser.addoption('--dtypetest', action="store", default="float32")
