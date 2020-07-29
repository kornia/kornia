# Welcome to the Kornia setup.py.
#

import os
from setuptools import setup, find_packages
import subprocess
import distutils.command.clean


################
# The variables below define the current version under
# development and the current pytorch supported verions.
# WARNING: Becareful and do not touch those variables,
# unless you are a maintainer. Otherwise, could brake
# the package backward compatibility.

# NOTE(maintainers): modify this variable each time you do a release

kornia_pt_dependencies = {
'0.3.1': '1.5.0',
'0.2.2': '1.4.0',
'0.1.4': '1.3.1',
'0.1.4': '1.3.0',
'0.1.4': '1.2.0',
}

version = '0.3.2'

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
elif sha != 'Unknown':
    version += '+' + sha[:7]
print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, 'kornia', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


# open readme file and remove logo
readme = open('README.rst').read()
long_description = '\n'.join(readme.split('\n')[7:])


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

requirements = [
    'numpy',
    'torch>=' + kornia_pt_dependencies[version],
]


if __name__ == '__main__':
    write_version_file()
    setup(
        # Metadata
        name=package_name,
        version=version,
        author='Edgar Riba',
        author_email='contact@kornia.org',
        url='https://github.com/kornia/kornia',
        description='Open Source Differentiable Computer Vision Library for PyTorch',
        long_description=long_description,
        license='Apache License 2.0',
        python_requires='>=3.6',

        # Test
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],

        # Package info
        packages=find_packages(exclude=('docs', 'test', 'examples',)),

        zip_safe=True,
        install_requires=requirements,
        classifiers=[
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3 :: Only',
            'License :: OSI Approved :: Apache Software License',
            'Topic :: Scientific/Engineering :: Image Recognition',
            'Topic :: Software Development :: Libraries',
        ],
    )
