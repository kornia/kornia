# Welcome to the Kornia setup.py.
import sys

# Make sure that kornia is running on Python 3.7.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)

if sys.version_info < (3, 7, 0):
    raise RuntimeError("Kornia requires Python 3.7.0 or later.")


from setuptools import setup

setup(
    install_requires=['packaging', 'torch>=1.9.1'],
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    extras_require={
        'dev': [
            'isort',
            'kornia-rs==0.0.8',
            'mypy[reports]',
            'numpy',
            'opencv-python',
            'pre-commit>=2.0',
            'pydocstyle',
            'pytest==7.2.1',
            'pytest-cov==4.0.0',
            'scipy',
        ],
        'docs': [
            'PyYAML>=5.1,<6.1.0',
            'furo',
            'matplotlib',
            'opencv-python',
            'sphinx>=4.0',
            'sphinx-autodoc-defaultargs',
            'sphinx-autodoc-typehints',
            'sphinx-copybutton>=0.3',
            'sphinx-design',
            'sphinx-rtd-theme>0.5',
            'sphinxcontrib-bibtex',
            'sphinxcontrib-gtagjs',
            'sphinxcontrib-youtube',
            'torchvision',
        ],
        'x': ['accelerate==0.15.0'],
    },
)
