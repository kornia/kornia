# Welcome to the Kornia setup.py.
#
import re
import sys

# Make sure that kornia is running on Python 3.6.0 or later
# (to avoid running into this bug: https://bugs.python.org/issue29246)

if sys.version_info < (3, 6, 0):
    raise RuntimeError("Kornia requires Python 3.6.0 or later.")


from setuptools import find_packages, setup


def find_version(file_path: str) -> str:
    version_file = open(file_path).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if not version_match:
        raise RuntimeError(f"Unable to find version string in {file_path}")
    return version_match.group(1)


VERSION = find_version("kornia/_version.py")


# NOTE: kornia MUST only require PyTorch
requirements = ['torch>=1.8.1', 'packaging']

# open readme file and set long description
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()


def load_requirements(filename: str):
    with open(filename) as f:
        return [x.strip() for x in f.readlines() if "-r" != x[0:2]]


requirements_extras = {"x": load_requirements("requirements/x.txt"), "dev": load_requirements("requirements/dev.txt")}
requirements_extras["all"] = requirements_extras["x"] + requirements_extras["dev"]


if __name__ == '__main__':
    setup(
        name='kornia',
        version=VERSION,
        author='Edgar Riba',
        author_email='edgar@kornia.org',
        url='https://www.kornia.org',
        download_url='https://github.com/kornia/kornia',
        license='Apache License 2.0',
        description='Open Source Differentiable Computer Vision Library for PyTorch',
        long_description=long_description,
        long_description_content_type='text/markdown',
        python_requires='>=3.6',
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        packages=find_packages(exclude=('docs', 'test', 'examples')),
        package_data={"kornia": ["py.typed"]},
        data_files=[('', ['requirements/x.txt', 'requirements/dev.txt'])],
        zip_safe=True,
        install_requires=requirements,
        extras_require=requirements_extras,
        keywords=['computer vision', 'deep learning', 'pytorch'],
        project_urls={
            "Bug Tracker": "https://github.com/kornia/kornia/issues",
            "Documentation": "https://kornia.readthedocs.io/en/latest",
            "Source Code": "https://github.com/kornia/kornia",
        },
        classifiers=[
            'Environment :: GPU',
            'Environment :: Console',
            'Natural Language :: English',
            # How mature is this project? Common values are
            #   3 - Alpha, 4 - Beta, 5 - Production/Stable
            'Development Status :: 4 - Beta',
            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Information Technology',
            'Topic :: Software Development :: Libraries',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Image Processing',
            # Pick your license as you wish
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
    )
