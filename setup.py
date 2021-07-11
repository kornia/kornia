# Welcome to the Kornia setup.py.
#
import distutils.command.clean
import glob
import os
import shutil
import subprocess
import sys

from setuptools import find_packages, setup

################
# The variables below define the current version under
# development and the current pytorch supported verions.
# WARNING: Becareful and do not touch those variables,
# unless you are a maintainer. Otherwise, could brake
# the package backward compatibility.

# NOTE(maintainers): modify this variable each time you do a release

version = '0.5.6'  # this a tag for the current development version


# NOTE(maintainers): update this dictionary each time you do a release
# When multiple pytorch versions are associated with a single version of kornia,
# the oldest one is the requirement. The versions should be inequalities.
# Once a pytorch version (in the future) breaks a kornia version, we could just
# add a maximal version.
kornia_pt_dependencies = {
    '0.5.6': '>=1.6.0',
    '0.5.5': '>=1.6.0',
    '0.5.4': '>=1.6.0',
    '0.5.3': '>=1.6.0',
    '0.5.2': '>=1.6.0',
    '0.5.1': '>=1.6.0',
    '0.5.0': '>=1.6.0',
    '0.4.2': '>=1.5.1',
    '0.4.1': '>=1.6.0',
    '0.4.0': '>=1.6.0,<1.7.0',
    '0.3.2': '>=1.5.0,<1.6.0',
    '0.3.1': '>=1.5.0',
    '0.2.2': '>=1.4.0',
    '0.1.4': '>=1.2.0',
}


# version can be overiden eg with KORNIA_BUILD_VERSION so we map each possible kornia version to the dictionary keys
def dep_version(version):
    compatible_versions = [v for v in kornia_pt_dependencies if v >= version]
    compatible_versions += [sorted(kornia_pt_dependencies)[-1]]
    return min(compatible_versions)


#################################

sha = 'Unknown'
package_name = 'kornia'

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

if os.getenv('KORNIA_BUILD_VERSION'):
    version = os.getenv('KORNIA_BUILD_VERSION')
elif os.getenv('KORNIA_RELEASE'):
    pass
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, 'kornia', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


# open readme file and set long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class clean(distutils.command.clean.clean):
    def run(self):
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split('\n')):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)

    # remove compiled and temporary files
    subprocess.call(['rm -rf dist/ build/ kornia.egg*'], shell=True)


requirements = ['torch' + kornia_pt_dependencies[dep_version(version)]]


if __name__ == '__main__':
    write_version_file()
    setup(
        # Metadata
        name=package_name,
        version=version,
        author='Edgar Riba',
        author_email='contact@kornia.org',
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
        zip_safe=True,
        install_requires=requirements,
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
