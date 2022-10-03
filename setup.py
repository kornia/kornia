# Welcome to the Kornia setup.py.
import sys

# Make sure that kornia is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)

if sys.version_info < (3, 6, 0):
    raise RuntimeError("Kornia requires Python 3.6.0 or later.")


from setuptools import setup

setup()
