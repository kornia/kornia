#! /usr/bin/env python
from subprocess import run, CalledProcessError
import sys


def main():
    try:
        print("Typechecker (mypy)", flush=True)
        files_to_analyse = ['./torchgeometry/image/gaussian.py']
        for file_to_analyse in files_to_analyse:
            run("mypy " + file_to_analyse + " --ignore-missing-imports", shell=True, check=True)
        print("mypy checks passed")
    except CalledProcessError:
        sys.exit(1)

if __name__ == '__main__':
    main()
