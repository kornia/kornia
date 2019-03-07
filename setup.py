# Welcome to the PyTorch setup.py.
#
# Environment variables you are probably interested in:
#
# Environment variables for feature toggles:
#
#   PYTORCH_BUILD_VERSION
#   PYTORCH_BUILD_NUMBER
#     specify the version of PyTorch, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution

from __future__ import print_function
from setuptools import setup, find_packages
import subprocess
import os


# Constant known variables used throughout this file
cwd = os.path.dirname(os.path.abspath(__file__))


################################################################################
# Version, create_version_file, and package_name
#
# Example for release (0.1.2):
#  TORCHGEOMETRY_BUILD_VERSION=0.1.2 \
#  TORCHGEOMETRY_BUILD_NUMBER=1 python setup.py install
################################################################################
package_name = os.getenv('TORCHGEOMETRY_PACKAGE_NAME', 'torchgeometry')
version = '0.1.2'  # NOTE: modify this variable each time we do a release
if os.getenv('TORCHGEOMETRY_BUILD_VERSION'):
    assert os.getenv('TORCHGEOMETRY_BUILD_NUMBER') is not None
    build_number = int(os.getenv('TORCHGEOMETRY_BUILD_NUMBER'))
    version = os.getenv('TORCHGEOMETRY_BUILD_VERSION')
    if build_number > 1:
        version += '.post' + str(build_number)
else:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except Exception:
        pass
print("Building wheel {}-{}".format(package_name, version))


# all the work we need to do _before_ setup runs
def build_deps():
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'torchgeometry', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


# open readme file and remove logo
readme = open('README.rst').read()
long_description = '\n'.join(readme.split('\n')[7:])


requirements = [
    'torch>=1.0.0',
]


if __name__ == '__main__':
    build_deps()
    setup(
        # Metadata
	name=package_name,
	version=version,
	author='Edgar Riba',
	author_email='edgar.riba@gmail.com',
	url='https://github.com/arraiyopensource/torchgeometry',
	description='differential geometric computer vision for deep learning',
	long_description=long_description,
	license='BSD',

	# Test
	setup_requires=['pytest-runner'],
	tests_require=['pytest'],

	# Package info
	packages=find_packages(exclude=('docs', 'test', 'examples',)),

	zip_safe=True,
	install_requires=requirements,
    )
