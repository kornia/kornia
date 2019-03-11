import pytest


def pytest_addoption(parser):
    parser.addoption('--typetest', action="store", default="cpu")
